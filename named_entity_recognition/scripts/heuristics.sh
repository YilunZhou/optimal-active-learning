#!/bin/bash
python heuristics.py --criterion min-confidence --evaluation-set valid
python heuristics.py --criterion normalized-min-confidence --evaluation-set valid
python heuristics.py --criterion longest --evaluation-set valid

python heuristics.py --criterion min-confidence --evaluation-set test
python heuristics.py --criterion normalized-min-confidence --evaluation-set test
python heuristics.py --criterion longest --evaluation-set test
