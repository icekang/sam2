#!/bin/bash

source activate sam2
which pythone

python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_s_MOSE_finetune_simple_optimizer.yaml \
    --use-cluster 0 \
    --num-gpus 2