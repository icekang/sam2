#!/bin/bash

source activate sam2
which python

export nnUNet_preprocessed=/home/gridsan/amanicka/datasets/nnUNet_Datasets/nnUNet_preprocessed
export nnUNet_raw=/home/gridsan/amanicka/datasets/nnUNet_Datasets/nnUNet_raw

python -u evaluate.py \
 --prompter_names mask scribble_2 \
 --fold 0 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask scribble_2 \
 --fold 1 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask scribble_2 \
 --fold 2 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask scribble_2 \
 --fold 3 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask scribble_2 \
 --fold 4 \
 --mask_every_n 16