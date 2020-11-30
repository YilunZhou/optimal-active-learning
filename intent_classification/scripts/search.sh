#!/bin/bash

# Base:
python search.py --model lstm --model-seed 0

# Model seed transfer:
python search.py --model lstm --model-seed 1
python search.py --model lstm --model-seed 2
python search.py --model lstm --model-seed 3
python search.py --model lstm --model-seed 4

# Architecture transfer:
python search.py --model cnn --model-seed 0
python search.py --model aoe --model-seed 0
python search.py --model roberta --model-seed 0
