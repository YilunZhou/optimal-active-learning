## Generate optimal orders for different model seeds
The command below will generate the log files in `logs/` directory.
```bash
sh scripts/search.sh
```

During the search, the following command summarizes the search logs.
```bash
python summarize.py logs
```

## Generate max-entropy, BALD and BatchBALD heuristics
BALD and BatchBALD heuristics are implemented by [`batchbald_redux`](https://github.com/BlackHC/batchbald_redux/tree/master/batchbald_redux) package. Source code is included in `batchbald_redux/` directory for convenience.

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

Plots will be saved in `../figures/object_classification/perf_curves/` directory.
```bash
sh scripts/performance_curve.sh
```

## Calculate and plot the model transfer matrix
The command will use results in `statistics/curves.shv` to plot the seed transfer matrix.

A plot named `../figures/object_classification/seed_transfer.pdf` will be saved.
```bash
python seeds_transfer.py
```

## Visualize the input (t-SNE embedding) and output (label) distributions
Three plot named `distribution_vis_max-entropy.pdf`, `distribution_vis_bald.pdf` `distribution_vis_batchbald.pdf` will be saved to `../figures/object_classification/` directory.
```bash
sh scripts/distribution_vis.sh
```

## Run Distribution-Matching Regularization (DMR)
### Input Distribution-Matching Regularization (IDMR)
A plot named `../figures/object_classification/idmr.pdf` will be saved.
```bash
python idmr.py
```
### Output Distribution-Matching Regularization (ODMR)
A plot named `../figures/object_classification/odmr.pdf` will be saved.
```bash
python odmr.py
```

## Compile and print optimal, heuristic, and random quality numbers
```bash
python collect_results.py
```
