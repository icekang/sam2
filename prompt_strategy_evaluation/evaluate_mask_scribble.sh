#!/bin/bash

source activate sam2
which python

python -u evaluate.py \
 --prompter_names mask scribble \
 --fold 0 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask scribble \
 --fold 1 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask scribble \
 --fold 2 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask scribble \
 --fold 3 \
 --mask_every_n 16

python -u evaluate.py \
 --prompter_names mask scribble \
 --fold 4 \
 --mask_every_n 16