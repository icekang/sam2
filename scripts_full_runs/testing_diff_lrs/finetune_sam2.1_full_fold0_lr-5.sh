#!/bin/bash

source activate sam2
which pythone

python training/train.py \
    -c configs/sam2.1_training/splits_final/fold0_lr-5.yaml \
    --use-cluster 0 \
    --num-gpus 2