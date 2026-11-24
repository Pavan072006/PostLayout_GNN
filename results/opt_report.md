# Post-layout-aware sizing (project 1 optimiser idea + project 4 GNN)

Each design is optimised for minimum power under the spec, then laid out with gLayout, extracted with Magic and re-simulated (typical corner).

| spec | method | seed | GBW optimiser saw [MHz] | GBW after layout [MHz] | GBW min | PM after layout | power [uW] | LVS | meets spec after layout |
|---|---|---|---|---|---|---|---|---|---|
| P1 | schematic only | 0 | 60.1 | 59.4 | 60 | 78.2 | 78 | match | **no** |
| P1 | schematic only | 1 | 60.2 | 57.8 | 60 | 77.8 | 83 | match | **no** |
| P1 | schematic + 5% guard band | 0 | 63.1 | 61.5 | 60 | 76.4 | 83 | match | yes |
| P1 | schematic + 5% guard band | 1 | 63.1 | 61.1 | 60 | 76.3 | 89 | match | yes |
| P1 | schematic + GNN parasitics | 0 | 60.0 | 60.1 | 60 | 78.1 | 78 | match | yes |
| P1 | schematic + GNN parasitics | 1 | 60.2 | 60.2 | 60 | 77.2 | 85 | match | yes |
| P2 | schematic only | 0 | 100.1 | 98.6 | 100 | 70.4 | 70 | match | **no** |
| P2 | schematic only | 1 | 100.0 | 96.9 | 100 | 73.2 | 81 | match | **no** |
| P2 | schematic + 5% guard band | 0 | 105.3 | 103.0 | 100 | 73.5 | 72 | match | yes |
| P2 | schematic + 5% guard band | 1 | 105.1 | 101.7 | 100 | 73.1 | 85 | match | yes |
| P2 | schematic + GNN parasitics | 0 | 100.8 | 100.9 | 100 | 74.1 | 73 | match | yes |
| P2 | schematic + GNN parasitics | 1 | 100.4 | 100.4 | 100 | 74.0 | 74 | match | yes |
| P3 | schematic only | 0 | 40.0 | 40.1 | 40 | 79.3 | 84 | match | yes |
| P3 | schematic only | 1 | 40.1 | 39.9 | 40 | 78.9 | 87 | match | **no** |
| P3 | schematic + 5% guard band | 0 | 42.0 | 41.3 | 40 | 80.2 | 89 | match | yes |
| P3 | schematic + 5% guard band | 1 | 42.0 | 42.0 | 40 | 78.4 | 91 | match | yes |
| P3 | schematic + GNN parasitics | 0 | 40.0 | 40.1 | 40 | 79.1 | 84 | match | yes |
| P3 | schematic + GNN parasitics | 1 | 40.0 | 40.0 | 40 | 80.4 | 90 | match | yes |

| method | designs meeting spec after layout | mean power | mean |predicted - real| GBW |
|---|---|---|---|
| schematic only | 1/6 | 16.1 % of budget | 1.72 % |
| schematic + 5% guard band | 6/6 | 16.9 % of budget | 2.16 % |
| schematic + GNN parasitics | 6/6 | 16.2 % of budget | 0.08 % |
