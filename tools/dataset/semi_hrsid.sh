#!/usr/bin/env bash
set -x
data_root=data
OFFSET=$RANDOM
for percent in 1 5 10; do
    for fold in 1 2 3 4 5; do
        python tools/dataset/semi_hrsid.py --percent ${percent} --seed ${fold} --data-dir "${data_root}"/HRSID --seed-offset ${OFFSET}
    done
done
