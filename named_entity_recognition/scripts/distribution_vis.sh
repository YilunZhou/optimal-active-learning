#!/bin/bash
python distribution_vis.py --criterion min-confidence
python distribution_vis.py --criterion normalized-min-confidence
python distribution_vis.py --criterion longest-min-confidence
