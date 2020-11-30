#!/bin/bash
python random_baselines.py --evaluation-set valid --model-seed 0
python random_baselines.py --evaluation-set valid --model-seed 1
python random_baselines.py --evaluation-set valid --model-seed 2
python random_baselines.py --evaluation-set valid --model-seed 3
python random_baselines.py --evaluation-set valid --model-seed 4

python random_baselines.py --evaluation-set test --model-seed 0
python random_baselines.py --evaluation-set test --model-seed 1
python random_baselines.py --evaluation-set test --model-seed 2
python random_baselines.py --evaluation-set test --model-seed 3
python random_baselines.py --evaluation-set test --model-seed 4
