#!/bin/bash

source activate sam2
which python

python -u evaluate.py \
 --prompter_names mask k_consistent_point \
 --fold 0