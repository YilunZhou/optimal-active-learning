#!/bin/bash
python distribution_vis.py --criterion max-entropy
python distribution_vis.py --criterion bald
python distribution_vis.py --criterion batchbald
