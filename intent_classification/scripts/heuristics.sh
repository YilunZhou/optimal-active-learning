#!/bin/bash

python heuristics.py --criterion max-entropy --evaluation-set valid --model lstm --model-seed 0
python heuristics.py --criterion max-entropy --evaluation-set valid --model lstm --model-seed 1
python heuristics.py --criterion max-entropy --evaluation-set valid --model lstm --model-seed 2
python heuristics.py --criterion max-entropy --evaluation-set valid --model lstm --model-seed 3
python heuristics.py --criterion max-entropy --evaluation-set valid --model lstm --model-seed 4
python heuristics.py --criterion max-entropy --evaluation-set valid --model cnn --model-seed 0
python heuristics.py --criterion max-entropy --evaluation-set valid --model aoe --model-seed 0
python heuristics.py --criterion max-entropy --evaluation-set valid --model roberta --model-seed 0

python heuristics.py --criterion max-entropy --evaluation-set test --model lstm --model-seed 0
python heuristics.py --criterion max-entropy --evaluation-set test --model lstm --model-seed 1
python heuristics.py --criterion max-entropy --evaluation-set test --model lstm --model-seed 2
python heuristics.py --criterion max-entropy --evaluation-set test --model lstm --model-seed 3
python heuristics.py --criterion max-entropy --evaluation-set test --model lstm --model-seed 4
python heuristics.py --criterion max-entropy --evaluation-set test --model cnn --model-seed 0
python heuristics.py --criterion max-entropy --evaluation-set test --model aoe --model-seed 0
python heuristics.py --criterion max-entropy --evaluation-set test --model roberta --model-seed 0

python heuristics.py --criterion bald --evaluation-set valid --model lstm --model-seed 0
python heuristics.py --criterion bald --evaluation-set valid --model lstm --model-seed 1
python heuristics.py --criterion bald --evaluation-set valid --model lstm --model-seed 2
python heuristics.py --criterion bald --evaluation-set valid --model lstm --model-seed 3
python heuristics.py --criterion bald --evaluation-set valid --model lstm --model-seed 4
python heuristics.py --criterion bald --evaluation-set valid --model cnn --model-seed 0
python heuristics.py --criterion bald --evaluation-set valid --model aoe --model-seed 0
python heuristics.py --criterion bald --evaluation-set valid --model roberta --model-seed 0

python heuristics.py --criterion bald --evaluation-set test --model lstm --model-seed 0
python heuristics.py --criterion bald --evaluation-set test --model lstm --model-seed 1
python heuristics.py --criterion bald --evaluation-set test --model lstm --model-seed 2
python heuristics.py --criterion bald --evaluation-set test --model lstm --model-seed 3
python heuristics.py --criterion bald --evaluation-set test --model lstm --model-seed 4
python heuristics.py --criterion bald --evaluation-set test --model cnn --model-seed 0
python heuristics.py --criterion bald --evaluation-set test --model aoe --model-seed 0
python heuristics.py --criterion bald --evaluation-set test --model roberta --model-seed 0
