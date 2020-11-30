#!/bin/bash
python search.py --model-seed 0
# additional search for studying seed transfer behavior
python search.py --model-seed 1
python search.py --model-seed 2
python search.py --model-seed 3
python search.py --model-seed 4
