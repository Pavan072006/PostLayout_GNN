#!/bin/bash
cd /foss/designs/postlayout_gnn
mkdir -p logs
PY="/foss/designs/glenv/bin/python -E"
$PY gen/gen_ota5t.py --n ${N_OTA:-400} --workers ${W_OTA:-4} --seed 1 >> logs/gen_ota5t.log 2>&1 &
$PY gen/gen_ts2.py   --n ${N_TS2:-300} --workers ${W_TS2:-3} --seed 2 >> logs/gen_ts2.log 2>&1 &
wait
