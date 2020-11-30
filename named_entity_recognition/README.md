## Generate the optimal order
The command below will generate the log file in `logs/` directory.
```bash
python search.py
```

During the search, the following command summarizes the search logs.
```bash
python summarize.py logs
```

## Generate least-confidence, normalized-least-confidence, and longest heuristics
The command below will write the curves and orders to `statistics/baselines.shv`.
```bash
sh scripts/heuristics.sh
```

## Generate random baselines
The command below will write the averaged curves to `statistics/baselines.shv`.
```bash
sh scripts/random_baselines.sh
```

## Plot the performance curves for optimal, heuristic, and random baselines
The command below will write the curves on validation and test sets to `statistics/curves.shv`.

A plot named `../figures/named_entity_recognition/perf_curves/s0_e0.pdf` will be saved.
```bash
python performance_curve.py
```

## Visualize the input (length) and output (label) distributions
Three plot named `distribution_vis_min-confidence.pdf`, `distribution_vis_normalized-min-confidence.pdf` `distribution_vis_longest.pdf` will be saved to `../figures/named_entity_recognition/` directory.
```bash
sh scripts/distribution_vis.sh
```

## Run Distribution-Matching Regularization (DMR)
### Input Distribution-Matching Regularization (IDMR)
Three plots named `idmr_min-confidence.pdf`, `idmr_normalized-min-confidence.pdf` `idmr_longest.pdf` will be saved to  `../figures/named_entity_recognition/` directory.
```bash
sh scripts/idmr.sh
```
### Output Distribution-Matching Regularization (ODMR)
Since the output is structured (i.e. correlated tags of a sentence), it is not straightforward to regularize the tag distribution. Thus, ODMR on this task is left to future work.

## Compile and print optimal, heuristic, and random quality numbers
```bash
python collect_results.py
```
