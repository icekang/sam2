#!/bin/bash

source activate sam2
which python
#10 prompts
python -u evaluate.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --k_consistent_point_k 1 \
 --k_border_3_pos_k 2 \
 --k_border_3_neg_k 4 \
 --fold 0 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --k_consistent_point_k 1 \
 --k_border_3_pos_k 2 \
 --k_border_3_neg_k 4 \
 --fold 1 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --k_consistent_point_k 1 \
 --k_border_3_pos_k 2 \
 --k_border_3_neg_k 4 \
 --fold 2 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --k_consistent_point_k 1 \
 --k_border_3_pos_k 2 \
 --k_border_3_neg_k 4 \
 --fold 3 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --k_consistent_point_k 1 \
 --k_border_3_pos_k 2 \
 --k_border_3_neg_k 4 \
 --fold 4 \
 --mask_every_n 16