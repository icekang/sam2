#!/bin/bash

source activate sam2
which python

python -u evaluate.py \
 --prompter_names mask k_consistent_point \
 --fold 0 \
 --model_cfg_name sam2.1_hiera_s_MOSE_finetune.yaml