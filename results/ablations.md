# Ablations

(GNN: ensemble of 2 seeds, 1000 epochs; metrics on signal-net total caps and on all device-geometry values)

## A. Message-passing depth

| K | net cap MAPE % | net cap R² | geometry MAPE % | geometry R² |
|---|---|---|---|---|
| K=0 | 44.6 | 0.236 | 11.1 | 0.916 |
| K=1 | 1.7 | 0.998 | 3.3 | 0.997 |
| K=2 | 1.6 | 0.998 | 3.3 | 0.997 |
| K=3 | 1.5 | 0.998 | 2.8 | 0.997 |

## B. Topology transfer: train on 5T OTA, test on 260 two-stage OTAs

| setting | net cap MAPE % | net cap R² | geometry MAPE % | geometry R² |
|---|---|---|---|---|
| zero-shot GNN | 23.0 | 0.859 | 18.3 | 0.731 |
| zero-shot GBM | 21.9 | 0.808 | 16.5 | 0.785 |
| few-shot (+20 ts2) GNN | 3.8 | 0.992 | 5.2 | 0.994 |
| few-shot (+20 ts2) GBM | 6.3 | 0.971 | 9.9 | 0.907 |

## C. Learning curve (same test set)

| train graphs | GNN net MAPE % | GBM net MAPE % | GNN geo MAPE % | GBM geo MAPE % |
|---|---|---|---|---|
| 122 | 2.0 | 5.2 | 3.4 | 7.4 |
| 245 | 1.6 | 4.3 | 3.0 | 6.5 |
| 491 | 1.5 | 4.0 | 2.7 | 5.8 |
