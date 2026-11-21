# Notes: post-layout parasitic prediction with a GNN (plain-language explanations)

These notes explain the ideas behind this project so you can explain them in an interview.
Numbers quoted here come from `results/results.md` and `results/ablations.md`.

---

## 1. The problem: the schematic-to-layout gap

A schematic is an idealised circuit: transistors joined by perfect wires. A real chip has
**layout**: metal wires with area, wires running next to each other, and transistor drains and
sources that are physical patches of silicon. All of that adds **parasitic capacitance**
(and resistance) that the schematic never sees. So the circuit you simulated before layout
("pre-layout") is faster and more stable than what you actually get ("post-layout").

The normal flow is slow: design the schematic, draw the layout, extract parasitics, re-simulate,
find the specs are missed, go back. Every loop costs hours to days. If a model can predict
the parasitics **from the schematic alone**, the designer can add them to the pre-layout sim
and catch the problem before drawing anything. That is what this project does.

## 2. What PEX is

**PEX = parasitic extraction.** A tool (here **Magic**, the open-source layout editor) reads
the layout's geometry and writes a netlist that contains:

* the transistors **as drawn**: real finger widths, and real drain/source areas (AD, AS) and
  perimeters (PD, PS), which set the junction (diode) capacitances, and
* extra capacitors: each wire's cap **to the substrate** (the "ground" plane under everything)
  and **coupling** caps between neighbouring nets.

Magic computes caps from the area and side length of each metal shape, using per-layer
coefficients from the PDK. We extract C only (no wire R). In this size of analog block, wires
are short and wide, so R barely matters; it would for long nets or high frequencies (limitation).

One 5T OTA layout here has ~30 parasitic caps totalling ~100–300 fF. The signal nets carry
roughly 5–60 fF each.

## 3. Why parasitics shift GBW and phase margin

**5-transistor OTA** (NMOS pair, PMOS mirror load):
* GBW ≈ gm1 / (2π·C_out). C_out = CL + whatever parasitic cap sits on OUT. More cap on OUT
  means lower GBW. With a 2 pF load, 10 fF is 0.5%. With 100 fF loads it is 10%, so small
  loads make layout matter a lot. That's why we swept CL from 0.1 to 2 pF.
* The mirror node X has a pole at ≈ gm3 / C_X. C_X is the gate caps of M3+M4 plus the
  junctions plus the wiring. This "mirror pole" eats phase margin: more C_X means lower PM.
* The tail node only affects common-mode behaviour, so it barely matters for the open-loop AC.

**Two-stage Miller OTA**:
* GBW ≈ gm1 / Cc. Any coupling cap between OUT1 and OUT adds directly to Cc, so GBW drops.
* The output pole p2 ≈ gm7 / (CL + C_OUT) and the OUT1 node cap also move the non-dominant
  poles, which changes PM.

So a parasitic's position matters as much as its size: 5 fF on OUT1–OUT (Miller) does more
than 5 fF on VDD.

## 4. What we found about where the shift comes from (key insight)

We ran four simulations of every layout:

| sim | what it contains |
|---|---|
| pre | finger-accurate schematic (each transistor split into the same fingers as the layout), default junction estimate |
| oracle caps | pre + the true extracted wiring caps |
| oracle caps+geo | pre + true caps + true junction geometry (AD/AS/PD/PS) from the layout |
| post | the full extracted netlist |

Result: **"oracle caps+geo" reproduces "post" to within 0.01%.** Wiring caps alone leave a
real error. Part of the wiring-cap effect is cancelled by smaller junctions: in the layout,
neighbouring fingers *share* a drain or source diffusion. That makes the real junction areas
smaller than the schematic's default "every finger has its own diffusion" estimate.

So the post-layout shift splits exactly into two parts we can learn:
1. **per-net wiring capacitance** (net-level and net-pair-level targets), and
2. **per-transistor junction geometry** (device-level targets).

This is the same split ParaGraph used: net parasitics plus device (layout-dependent) parameters.

One more trap we hit: our first "pre" sim modelled each transistor as one wide finger.
The layout uses fingers of 0.5–8 µm, and the transistor model has narrow-width effects, so gain
and gm differed even with no parasitics at all. Real schematics carry the finger count (nf), so
we fixed the reference to be finger-accurate. Otherwise the "layout shift" would include an
effect that is really just a different transistor shape.

### Headline numbers (memorise these)
* Dataset: 702 layouts (402 5T OTA + 300 two-stage), 33 LVS spot-checks, all matched.
* Physics: true caps alone miss the GBW shift by 0.88 %-pts; caps + junction geometry miss it by 0.03.
* Per-net cap on the test set: GNN 1.3 % MAPE (0.22 fF), GBM 4.0 %, ridge 39 %, type-mean 61 %.
* Junction geometry: GNN ~2.4 % MAPE vs GBM ~5.8 % vs the schematic default ~26 %.
* GBW shift (true −2.8 ± 2.1 %): sim + GNN caps + geometry gives 0.15 %-pts MAE (R² 0.98), PM 0.06°
  (R² 0.96). Ignoring layout: 2.8 %-pts error.
* K = 0 (no message passing): 45 % MAPE; K = 1: 1.7 %. The graph context matters.
* Zero-shot to an unseen topology: ~22 % for both models. With 20 examples: GNN 3.8 % vs GBM 6.3 %.

## 5. The netlist as a graph

A netlist is naturally a **bipartite graph**:
* **device nodes** (one per transistor). Features: NMOS/PMOS, W, L, finger width, number of
  fingers, gate area.
* **net nodes** (one per wire). Features: what the net does (input, output, stage-1 output,
  internal, bias, supply, ground), how many drains/gates/sources/bulks touch it, and whether
  it is a pin.
* **edges** connect a device to the nets on its terminals, labelled **drain / gate / source /
  bulk**. The label matters: a net touching 4 µm of *drain* gets junction-plus-wiring cap,
  while a net touching 4 µm of *gate* gets gate-routing cap. Those are different physics.

Everything in the graph is known from the schematic plus sizing plus the finger plan. Nothing
comes from the layout. Only the targets come from layout + PEX.

## 6. How the GNN message-passes over a netlist

Think of each node holding a small vector of numbers (an "embedding"), its current
summary of itself. One round of message passing:

1. every **device** sends a message to each net it touches. The message is a small neural
   net applied to (device embedding, terminal type). For example: "I'm a 40 µm PMOS and
   you're my drain."
2. every **net** adds up the messages it received and updates its own embedding.
3. then the nets send messages back to the devices the same way.

After 1 round, net X "knows" which transistors sit on it. After 2 rounds it knows about the
other nets those transistors touch (e.g. X learns M3 also touches VDD, so X is a mirror
node). After 3 rounds it also knows about the devices on those nets. That neighbourhood is
what sets layout size and routing length, which is what sets parasitics. A per-net model
with hand features only sees step 1.

**Heads** (the outputs):
* pair head: for every pair of nets (i, j), predict C(i, j) from [h_i + h_j, h_i · h_j, pair
  structure]. Writing it with "+" and "·" makes it symmetric: C(i, j) = C(j, i). "Pair
  structure" counts how many transistors the two nets share and on which terminals
  (gate–drain of the same device, for example).
* net total = sum of its predicted pair caps, so totals and pairs always agree.
* device head: predicts AD/AS/PD/PS per transistor.
* graph head: pools everything plus CL, Cc, Itail and the pre-layout sim, and predicts the GBW
  and PM shift directly.

We trained on log(1 + C) because caps span 0.01 fF to 200 fF. A plain squared error would only
care about the biggest caps.

The GNN is ~100 lines of plain PyTorch (`model/gnn.py`, no PyG), so each step can be explained.
An ensemble of a few seeds is averaged, which is cheap because the graphs are tiny.

## 7. Why baselines matter

A fancy model is only interesting if it beats simple ones. We compare:
* **type mean**: "every OUT net gets the average OUT cap". This is the "do nothing" number.
* **ridge**: linear regression on hand features (sum of W on drains, gate area, ...).
* **GBM**: gradient-boosted trees on the same hand features. This is a strong tabular model
  and often hard to beat on small data. The GBM pair model gets the *same* pair-structure
  features as the GNN, so any gap is due to message passing, not extra inputs.
* **schematic estimate** for junctions: the default formula the simulator would use.
* **zero shift**: "post = pre". The error you live with today if you ignore layout.

If the GNN does not beat GBM, the honest conclusion is "graphs don't help on this data". That
is still a useful result. Ablations check the reasons: depth K = 0 (no message passing),
topology transfer, and a learning curve.

## 8. ParaGraph (DAC 2020) in one paragraph

*Ren, Kokai, Turner, Ku (NVIDIA), "ParaGraph: Layout Parasitics and Device Parameter Prediction
using Graph Neural Networks", DAC 2020.* They turned large industrial analog/mixed-signal
schematics (FinFET node) into **heterogeneous graphs**: separate node types for nets and for
each kind of device, with edge types by terminal. They used a GNN layer that combines
GraphSAGE-style neighbour aggregation with attention. They predicted **net parasitic
capacitance** and **device layout parameters** (layout-dependent effects such as diffusion
geometry). Two tricks were important: an **ensemble of models trained on different value
ranges**, because caps span orders of magnitude, and feeding the predictions back into
**simulation** to show pre-layout sims got much closer to post-layout. They reported large
improvements over designer estimates. Check the paper for exact figures before quoting
numbers. This project is a small open-source replica of that idea: GF180 instead of FinFET,
generated layouts instead of industrial ones, and a hand-written bipartite GNN.

Differences to say out loud: they had thousands of real designs. We have a few hundred
generated layouts of two topologies, so our graphs are small, and a GBM with good features is
a much tougher competitor here than it would be on big, diverse circuits.

## 9. Pipeline (what each script does)

```
random sizing (gm/Id LUT) ─► gLayout layout (GF180) ─► Magic PEX ─► net matching ─► JSON sample
      gen/gen_ota5t.py, gen/gen_ts2.py         gen/pex_tools.py   gen/netlist_utils.py
                                                        │
                    gen/resim_dataset.py: finger-accurate pre / oracle sims (ngspice)
                                                        │
model/dataset.py: bipartite graphs ─► model/baselines.py, model/gnn.py ─► model/run_all.py
                                                     (metrics, re-simulation with predicted caps, plots)
```

**Net matching.** Magic names internal nets like `a_n2175_3456#`. We map them back to
schematic names (X, TAIL) by colour refinement on the device/net graph. It's the same idea as
the Weisfeiler–Lehman test that also explains what GNNs can and cannot tell apart: pins keep
their names, and an internal net is identified by "which kind of terminal of which kind of
transistor touches it", refined over a few rounds.

---

## 10. Likely interview questions (with short answers)

**Q: What problem does this solve?**
Pre-layout sims are optimistic because they miss parasitics. Waiting for layout + extraction
to find out costs days per iteration. We predict the parasitics from the schematic in
milliseconds and add them to the pre-layout sim, so the designer sees the post-layout GBW/PM
shift early.

**Q: Why a graph neural network and not a table model?**
A netlist *is* a graph, and a net's parasitic depends on its neighbourhood (which transistors,
how big, what else they connect to). A GNN learns that neighbourhood summary instead of us
hand-designing it. On our data the GNN beat a GBM with the same pair features. K = 0
(no message passing) is much worse, so the graph context is what helps.

**Q: How did you get training data?**
Generated it: random but sensible sizings from a gm/Id lookup table, auto-layout with gLayout
(GF180MCU), Magic parasitic extraction, ngspice before/after. About 1–2 min per sample, run in
parallel. Two topologies (5T OTA and two-stage Miller OTA). LVS was spot-checked on every
20th–25th sample and all matched.

**Q: What exactly does the model predict?**
(1) cap between every pair of nets, and hence each net's total cap; (2) each transistor's
junction geometry (AD/AS/PD/PS); (3) directly, the GBW and PM shift. (1) + (2) go into the
schematic sim to get the shift with a real simulator.

**Q: How do you know the predicted caps are "good enough"?**
Not only by MAPE: we put them back into the simulator and compare the GBW/PM shift with
the true post-layout sim. We also showed the ceiling: true caps + true geometry reproduce
post-layout exactly, so any remaining error is the model's.

**Q: Why did wiring caps alone not reproduce the post-layout shift?**
Because the layout's transistors differ from the schematic's: shared diffusion between fingers
makes junction caps smaller than the default estimate. That partly cancels the added wiring
cap, so you need both.

**Q: Why did you need a finger-accurate pre-layout schematic?**
Narrow fingers behave differently from one wide transistor (narrow-width effects in BSIM).
Without matching fingers, part of the "layout shift" is really a different device, not a
parasitic.

**Q: Why train on log(C)?**
Caps span about four orders of magnitude (0.01–200 fF). Squared error on raw fF would ignore all
the small coupling caps, and those matter (e.g. Miller coupling).

**Q: How did you make the pair prediction symmetric?**
The head only sees h_i + h_j and h_i · h_j, which don't change when i and j swap.

**Q: Does it generalise to a topology it never saw?**
Not zero-shot: trained only on the 5T OTA, it gets ~22 % error on the two-stage OTA (the GBM
is the same). The new stage and the new net role (OUT1) are out of distribution. With just 20
two-stage examples it drops to 3.8 % (GBM 6.3 %), so the GNN reuses what it learned about
the shared first stage. In practice: fine-tune on a few layouts of each new cell family.

**Q: The GBM is already at 4 %. Is the GNN worth it?**
On these small circuits the gain is in the details: coupling caps (3× lower error), junction
geometry (2× lower), and data efficiency (GNN with 1/4 of the data beats GBM with all of it).
Those details are what move the simulated GBW/PM: R² 0.98 vs 0.92 for the shift. On larger,
more varied circuits, hand features get harder to design, and that's where graphs should pay off
more.

**Q: What are the limitations?**
Only two topologies and one layout generator. The model learns gLayout's style, not a
human's. C only, no wire R. Typical corner only for the dataset. Small graphs, so a
well-featured GBM is close. A real deployment would need diverse, human-drawn layouts from
the target team and PDK.

**Q: How would you scale this to a real design team?**
Harvest (schematic, extracted netlist) pairs from past tape-outs. Use heterogeneous node types
per device family and add R and LDE targets. Train per PDK. Ship it as a pre-layout sim plug-in
that writes predicted parasitics into the schematic testbench, plus uncertainty (ensemble
spread) to flag nets that need a real extraction.

**Q: Why ngspice/Magic/gLayout?**
They're open source, so the whole flow is reproducible: GF180MCU is an open PDK.

**Q: How long does inference take vs extraction?**
Inference: milliseconds per circuit. Layout + extraction: ~1 min here, hours to days with a
human layout.

**Q: How did you check the auto-generated layouts are valid?**
Netgen LVS (layout vs schematic) on a sample of layouts. All matched. DRC was skipped for
speed in data generation (project 1 showed the same blocks are DRC-clean); the parasitics
don't depend on DRC fixes much.

**Q: What would you try next?**
More topologies (folded cascode, comparator, LDO). Predicting R. Uncertainty estimates.
Training on human layouts. Using the predictions inside the sizing loop of project 1, so the
optimiser sizes for post-layout performance directly.
