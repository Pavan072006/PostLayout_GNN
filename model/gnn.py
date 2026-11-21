import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

from dataset import DEV_FEAT_DIM, NET_FEAT_DIM, GLOB_FEAT_DIM, PAIR_FEAT_DIM


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def scatter_sum(src, idx, n):
    out = torch.zeros(n, src.shape[1], dtype=src.dtype)
    return out.index_add_(0, idx, src)


class Batch:

    def __init__(self, graphs, norm):
        xd, xn, ed, en, et, pi, pj, yp, yt, gd, gn, gp, gl, ygr, yg, pf = ([] for _ in range(16))
        od = on = 0
        for k, g in enumerate(graphs):
            nd, nn_ = len(g["x_dev"]), len(g["x_net"])
            xd.append(g["x_dev"]); xn.append(g["x_net"])
            ed.append(g["e_dev"] + od); en.append(g["e_net"] + on); et.append(g["e_t"])
            pi.append(g["pair_i"] + on); pj.append(g["pair_j"] + on); pf.append(g["pair_f"])
            yp.append(g["y_pair"]); yt.append(g["y_tot"])
            gd.append(np.full(nd, k)); gn.append(np.full(nn_, k)); gp.append(np.full(len(g["pair_i"]), k))
            gl.append(g["glob"]); ygr.append([g["dgbw"], g["dpm"]]); yg.append(g["y_geo"])
            od += nd; on += nn_
        f = lambda a, dt=torch.float32: torch.tensor(np.concatenate(a), dtype=dt)
        self.x_dev = (f(xd) - norm["dev_mu"]) / norm["dev_sd"]
        self.x_net = (f(xn) - norm["net_mu"]) / norm["net_sd"]
        self.glob = (torch.tensor(np.array(gl)) - norm["glob_mu"]) / norm["glob_sd"]
        L = torch.long
        self.e_dev, self.e_net, self.e_t = f(ed, L), f(en, L), f(et, L)
        self.pi, self.pj = f(pi, L), f(pj, L)
        self.pair_f = torch.log1p(f(pf))
        self.y_pair, self.y_tot, self.y_geo = f(yp), f(yt), f(yg)
        self.g_dev, self.g_net = f(gd, L), f(gn, L)
        self.y_graph = torch.tensor(ygr, dtype=torch.float32)
        self.n_dev, self.n_net, self.n_graph = od, on, len(graphs)


def make_norm(graphs):
    def ms(a):
        a = np.concatenate(a) if a[0].ndim == 2 else np.array(a)
        sd = a.std(0)
        sd[sd < 1e-6] = 1
        return torch.tensor(a.mean(0), dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)
    n = {}
    n["dev_mu"], n["dev_sd"] = ms([g["x_dev"] for g in graphs])
    n["net_mu"], n["net_sd"] = ms([g["x_net"] for g in graphs])
    n["glob_mu"], n["glob_sd"] = ms([g["glob"] for g in graphs])
    yg = np.array([[g["dgbw"], g["dpm"]] for g in graphs])
    n["yg_mu"], n["yg_sd"] = torch.tensor(yg.mean(0), dtype=torch.float32), torch.tensor(yg.std(0) + 1e-6, dtype=torch.float32)
    return n


class NetlistGNN(nn.Module):
    def __init__(self, hid=64, K=3):
        super().__init__()
        self.K = K
        self.dev_in = mlp(DEV_FEAT_DIM, hid, hid)
        self.net_in = mlp(NET_FEAT_DIM, hid, hid)
        self.term = nn.Embedding(4, hid)
        self.msg_dn = nn.ModuleList([mlp(2 * hid, hid, hid) for _ in range(K)])
        self.upd_n = nn.ModuleList([mlp(2 * hid, hid, hid) for _ in range(K)])
        self.msg_nd = nn.ModuleList([mlp(2 * hid, hid, hid) for _ in range(K)])
        self.upd_d = nn.ModuleList([mlp(2 * hid, hid, hid) for _ in range(K)])
        self.ln_n = nn.ModuleList([nn.LayerNorm(hid) for _ in range(K)])
        self.ln_d = nn.ModuleList([nn.LayerNorm(hid) for _ in range(K)])
        self.pair = mlp(2 * hid + PAIR_FEAT_DIM, hid, 1, n=3)
        self.graph = mlp(4 * hid + GLOB_FEAT_DIM, hid, 2, n=3)
        self.geo = mlp(hid, hid, 4, n=3)

    def forward(self, b):
        hd, hn = self.dev_in(b.x_dev), self.net_in(b.x_net)
        te = self.term(b.e_t)
        for k in range(self.K):
            m = self.msg_dn[k](torch.cat([hd[b.e_dev], te], 1))
            hn = self.ln_n[k](hn + self.upd_n[k](torch.cat([hn, scatter_sum(m, b.e_net, b.n_net)], 1)))
            m = self.msg_nd[k](torch.cat([hn[b.e_net], te], 1))
            hd = self.ln_d[k](hd + self.upd_d[k](torch.cat([hd, scatter_sum(m, b.e_dev, b.n_dev)], 1)))
        hi, hj = hn[b.pi], hn[b.pj]
        log_pair = self.pair(torch.cat([hi + hj, hi * hj, b.pair_f], 1)).squeeze(1)
        cap = torch.expm1(log_pair.clamp(max=8)).clamp(min=0)
        tot = torch.zeros(b.n_net).index_add_(0, b.pi, cap).index_add_(0, b.pj, cap)
        cnt_n = torch.zeros(b.n_graph).index_add_(0, b.g_net, torch.ones(b.n_net))[:, None]
        cnt_d = torch.zeros(b.n_graph).index_add_(0, b.g_dev, torch.ones(b.n_dev))[:, None]
        sn, sd = scatter_sum(hn, b.g_net, b.n_graph), scatter_sum(hd, b.g_dev, b.n_graph)
        pooled = torch.cat([sn / cnt_n, sd / cnt_d, sn / 10, sd / 10, b.glob], 1)
        return log_pair, tot, self.graph(pooled), self.geo(hd)


def loss_fn(out, b, norm, w_graph=0.3):
    log_pair, tot, yg, lgeo = out
    l_pair = ((log_pair - torch.log1p(b.y_pair)) ** 2).mean()
    l_tot = ((torch.log(tot + 0.1) - torch.log(b.y_tot + 0.1)) ** 2).mean()
    l_g = (((yg - (b.y_graph - norm["yg_mu"]) / norm["yg_sd"])) ** 2).mean()
    l_geo = ((lgeo - torch.log1p(b.y_geo)) ** 2).mean()
    return l_pair + l_tot + l_geo + w_graph * l_g, (l_pair.item(), l_tot.item(), l_geo.item(), l_g.item())


def train(train_g, val_g, seed=0, epochs=1500, lr=3e-3, hid=64, K=3, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    norm = make_norm(train_g)
    btr, bva = Batch(train_g, norm), Batch(val_g, norm)
    model = NetlistGNN(hid, K)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss, _ = loss_fn(model(btr), btr, norm)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vl, parts = loss_fn(model(bva), bva, norm)
            if vl.item() < best:
                best, best_state = vl.item(), {k: v.clone() for k, v in model.state_dict().items()}
            if verbose and ep % 200 == 0:
                print(f"  ep {ep:4d} train {loss.item():.4f} val {vl.item():.4f} parts {np.round(parts, 4)}")
    model.load_state_dict(best_state)
    model.eval()
    return model, norm


class Ensemble:

    def __init__(self, models_norms):
        self.mn = models_norms

    @torch.no_grad()
    def predict(self, graphs):
        P, T, G, Q = [], [], [], []
        for m, norm in self.mn:
            b = Batch(graphs, norm)
            lp, tot, yg, lgeo = m(b)
            Q.append(torch.expm1(lgeo).clamp(min=0).numpy())
            P.append(torch.expm1(lp.clamp(max=8)).clamp(min=0).numpy())
            T.append(tot.numpy())
            G.append((yg * norm["yg_sd"] + norm["yg_mu"]).numpy())
        P, T, G, Q = np.mean(P, 0), np.mean(T, 0), np.mean(G, 0), np.mean(Q, 0)
        # split back per graph
        out, op, on, od = [], 0, 0, 0
        for k, g in enumerate(graphs):
            npair, nnet = len(g["pair_i"]), len(g["nets"])
            ndev = len(g["devs"])
            out.append(dict(pair=P[op:op + npair], tot=T[on:on + nnet], geo=Q[od:od + ndev],
                            dgbw=G[k, 0], dpm=G[k, 1]))
            op += npair; on += nnet; od += ndev
        return out
