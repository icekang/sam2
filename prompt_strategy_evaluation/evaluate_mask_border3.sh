#!/bin/bash

source activate sam2
which python

python -u evaluate.py \
 --prompter_names mask k_border_3 \
 --fold 0 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask k_border_3 \
 --fold 1 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask k_border_3 \
 --fold 2 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask k_border_3 \
 --fold 3 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask k_border_3 \
 --fold 4 \
 --mask_every_n 16