#!/bin/bash

source activate sam2
which pythone

python training/train.py \
    -c configs/sam2.1_training/splits_final/fold4.yaml \
    --use-cluster 0 \
    --num-gpus 2