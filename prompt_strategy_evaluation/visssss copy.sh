#!/bin/bash

source activate sam2
which python

python -u visualize_SAM2_predictions.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --fold 0 \
 --mask_every_n 16

python -u visualize_SAM2_predictions.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --fold 1 \
 --mask_every_n 16

python -u visualize_SAM2_predictions.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --fold 2 \
 --mask_every_n 16

python -u visualize_SAM2_predictions.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --fold 3 \
 --mask_every_n 16


python -u visualize_SAM2_predictions.py \
 --prompter_names mask k_border_3 k_consistent_point \
 --fold 4 \
 --mask_every_n 16