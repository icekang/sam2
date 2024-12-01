#!/bin/bash

source activate sam2
which python

python -u evaluate_finetuned_sam2_negative_points.py
