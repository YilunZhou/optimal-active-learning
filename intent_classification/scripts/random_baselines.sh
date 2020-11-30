#!/bin/bash
python random_baselines.py --evaluation-set valid --model lstm --model-seed 0
python random_baselines.py --evaluation-set valid --model lstm --model-seed 1
python random_baselines.py --evaluation-set valid --model lstm --model-seed 2
python random_baselines.py --evaluation-set valid --model lstm --model-seed 3
python random_baselines.py --evaluation-set valid --model lstm --model-seed 4
python random_baselines.py --evaluation-set valid --model cnn --model-seed 0
python random_baselines.py --evaluation-set valid --model aoe --model-seed 0
python random_baselines.py --evaluation-set valid --model roberta --model-seed 0

python random_baselines.py --evaluation-set test --model lstm --model-seed 0
python random_baselines.py --evaluation-set test --model lstm --model-seed 1
python random_baselines.py --evaluation-set test --model lstm --model-seed 2
python random_baselines.py --evaluation-set test --model lstm --model-seed 3
python random_baselines.py --evaluation-set test --model lstm --model-seed 4
python random_baselines.py --evaluation-set test --model cnn --model-seed 0
python random_baselines.py --evaluation-set test --model aoe --model-seed 0
python random_baselines.py --evaluation-set test --model roberta --model-seed 0
