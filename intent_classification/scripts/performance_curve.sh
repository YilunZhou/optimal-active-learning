#!/bin/bash

# Pairwise LSTM seeds:
python performance_curve.py --model lstm --search-model-seed 0 --eval-model-seed 0
python performance_curve.py --model lstm --search-model-seed 0 --eval-model-seed 1
python performance_curve.py --model lstm --search-model-seed 0 --eval-model-seed 2
python performance_curve.py --model lstm --search-model-seed 0 --eval-model-seed 3
python performance_curve.py --model lstm --search-model-seed 0 --eval-model-seed 4

python performance_curve.py --model lstm --search-model-seed 1 --eval-model-seed 0
python performance_curve.py --model lstm --search-model-seed 1 --eval-model-seed 1
python performance_curve.py --model lstm --search-model-seed 1 --eval-model-seed 2
python performance_curve.py --model lstm --search-model-seed 1 --eval-model-seed 3
python performance_curve.py --model lstm --search-model-seed 1 --eval-model-seed 4

python performance_curve.py --model lstm --search-model-seed 2 --eval-model-seed 0
python performance_curve.py --model lstm --search-model-seed 2 --eval-model-seed 1
python performance_curve.py --model lstm --search-model-seed 2 --eval-model-seed 2
python performance_curve.py --model lstm --search-model-seed 2 --eval-model-seed 3
python performance_curve.py --model lstm --search-model-seed 2 --eval-model-seed 4

python performance_curve.py --model lstm --search-model-seed 3 --eval-model-seed 0
python performance_curve.py --model lstm --search-model-seed 3 --eval-model-seed 1
python performance_curve.py --model lstm --search-model-seed 3 --eval-model-seed 2
python performance_curve.py --model lstm --search-model-seed 3 --eval-model-seed 3
python performance_curve.py --model lstm --search-model-seed 3 --eval-model-seed 4

python performance_curve.py --model lstm --search-model-seed 4 --eval-model-seed 0
python performance_curve.py --model lstm --search-model-seed 4 --eval-model-seed 1
python performance_curve.py --model lstm --search-model-seed 4 --eval-model-seed 2
python performance_curve.py --model lstm --search-model-seed 4 --eval-model-seed 3
python performance_curve.py --model lstm --search-model-seed 4 --eval-model-seed 4

# Other architectures:
python performance_curve.py --model cnn --search-model-seed 0 --eval-model-seed 0
python performance_curve.py --model aoe --search-model-seed 0 --eval-model-seed 0
python performance_curve.py --model roberta --search-model-seed 0 --eval-model-seed 0
