#
"""
This script generates and prints a random sample of combinations from a predefined training list.

The script performs the following steps:
1. Imports necessary modules: `combinations` from `itertools` and `random`.
2. Sets a seed for the random number generator to ensure reproducibility.
3. Defines a list of training identifiers.
4. Generates all possible combinations of 4 elements from the training list.
5. Randomly samples 4 combinations from the generated list of combinations.
6. Prints each sampled combination, with each element of the combination on a new line.

Modules:
    itertools: Provides functions for creating iterators for efficient looping.
    random: Implements pseudo-random number generators for various distributions.

Functions:
    None

Variables:
    train_list (list): A list of training identifiers.
    combs (list): A list of all possible combinations of 4 elements from `train_list`.
    sampled_combs (list): A random sample of 4 combinations from `combs`.

Usage:
    Run the script to print a random sample of 4 combinations from the training list.
"""
from itertools import combinations
import random

random.seed(0)

train_list = ['101-019', '101-044', '106-002', '401-004', '701-013', '704-003']
combs = list(combinations(train_list, 4))

sampled_combs = random.sample(combs, 4)
for comb in sampled_combs:
    print('\n'.join(comb))
    print()