#!/bin/bash

source activate sam2
which python

python -u evaluate_finetuned_sam2_consistent_K_points_prompting.py
