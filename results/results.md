# Results

Samples: {'ota5t': 402, 'ts2': 300} (train/val/test = 491/105/106)

## Per-net total capacitance (test set, signal nets)

| model | MAE [fF] | MAPE [%] | R² |
|---|---|---|---|
| type_mean | 7.08 | 60.5 | 0.265 |
| ridge | 4.65 | 39.1 | 0.660 |
| gbm | 0.68 | 4.0 | 0.988 |
| gnn | 0.22 | 1.3 | 0.999 |
| gbm_pairs | 0.69 | 4.4 | 0.991 |

## MAPE [%] per net

| model | VP | VN | OUT | OUT1 | X | TAIL | IBIAS |
|---|---|---|---|---|---|---|---|
| type_mean | 34.5 | 34.5 | 66.4 | 28.5 | 58.4 | 54.2 | 128.5 |
| ridge | 44.6 | 44.9 | 71.5 | 30.9 | 28.2 | 17.7 | 31.0 |
| gbm | 2.0 | 2.1 | 6.7 | 4.8 | 3.7 | 5.5 | 3.9 |
| gnn | 0.5 | 0.8 | 2.6 | 1.7 | 1.2 | 1.0 | 1.8 |
| gbm_pairs | 4.0 | 3.5 | 6.6 | 4.6 | 3.4 | 4.5 | 4.1 |

## Pair (coupling) capacitance

| model | pairs | MAE [fF] | R² |
|---|---|---|---|
| gbm | all pairs | 0.458 | 0.961 |
| gbm | signal-signal coupling | 0.085 | 0.963 |
| gnn | all pairs | 0.083 | 1.000 |
| gnn | signal-signal coupling | 0.028 | 0.997 |

## Device junction geometry (AD/AS um², PD/PS um), MAPE [%] / R²

| model | AD | AS | PD | PS |
|---|---|---|---|---|
| schematic estimate | 37.4 / 0.843 | 21.3 / 0.889 | 35.3 / 0.767 | 10.5 / 0.848 |
| gbm | 7.6 / 0.931 | 4.5 / 0.947 | 6.8 / 0.941 | 4.1 / 0.954 |
| gnn | 3.1 / 0.996 | 2.6 / 0.996 | 2.1 / 0.998 | 1.7 / 0.998 |

## GBW / PM shift (post-layout minus pre-layout), test set

True shift on test set: dGBW = -2.77 ± 2.08 %, dPM = -0.76 ± 0.48 deg

| method | dGBW MAE [%-pts] | dGBW R² | dPM MAE [deg] | dPM R² |
|---|---|---|---|---|
| oracle caps only | 0.72 | 0.62 | 0.22 | 0.43 |
| oracle caps+geo | 0.04 | 1.00 | 0.01 | 0.99 |
| GNN direct | 0.26 | 0.96 | 0.13 | 0.81 |
| GBM direct | 0.59 | 0.85 | 0.23 | 0.44 |
| zero (pre=post) | 2.78 | -1.78 | 0.78 | -2.49 |
| sim+GNN caps (no geo) | 0.74 | 0.59 | 0.23 | 0.41 |
| sim+GNN caps+geo | 0.15 | 0.98 | 0.06 | 0.96 |
| sim+GBM caps+geo | 0.36 | 0.92 | 0.14 | 0.72 |
