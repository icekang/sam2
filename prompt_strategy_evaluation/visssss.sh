#!/bin/bash

source activate sam2
which python

python -u visualize_SAM2_predictions.py \
 --prompter_names mask scribble_2 \
 --fold 0 \
 --mask_every_n 16
