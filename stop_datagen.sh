#!/bin/bash
for p in $(ps -eo pid,args | grep -E "[g]en/gen_(ota5t|ts2)\.py|[r]un_datagen" | awk '{print $1}'); do
  pkill -9 -P $p 2>/dev/null; kill -9 $p 2>/dev/null
done
