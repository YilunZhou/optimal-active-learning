## Generate optimal orders for all architectures and different LSTM seeds
The command below will generate the log files in `logs/` directory.
```bash
sh scripts/search.sh
```

During the search, the following command summarizes the search logs.
```bash
python summarize.py logs
```

## Generate max-entropy and BALD heuristics for all architectures
The command below will write the curves and orders to `statistics/baselines.shv`.
```bash
sh scripts/heuristics.sh
```

## Generate random baselines for all architectures
The command below will write the averaged curves to `statistics/baselines.shv`.
```bash
sh scripts/random_baselines.sh
```

## Plot the performance curves for optimal, heuristic, and random baselines
The command below will write the curves on validation and test sets to `statistics/perf_curves.shv`.

Plots will be saved in `../figures/intent_classification/perf_curves/` directory.
```bash
sh scripts/performance_curve.sh
```

## Calculate and plot the seed transfer matrix
The command will use results in `statistics/perf_curves.shv` to plot the seed transfer matrix.

A plot named `../figures/intent_classification/seed_transfer.pdf` will be saved.
```bash
python seeds_transfer.py
```

## Calculate and plot the model transfer matrix
The command below will write the transfer qualities to `statistics/model_transfer.shv`.

A plot named `../figures/intent_classification/model_transfer.pdf` will be saved.
```bash
python model_transfer.py
```
## Visualize the relative order ranking between models
Plots will be saved in `../figures/intent_classification/relative_orders/` directory.
```bash
sh scripts/relative_order_ranking.sh
```

## Visualize the input (length) and output (label) distributions
A plot named `../figures/intent_classification/distribution_vis.pdf` will be saved.
```bash
python distribution_vis.py
```

## Run Distribution-Matching Regularization (DMR)
### Input Distribution-Matching Regularization (IDMR)
Two plots named `idmr_max-entropy.pdf` and `idmr_bald.pdf` will be saved to `../figures/intent_classification/` directory.
```bash
python idmr.py --criterion max-entropy
python idmr.py --criterion bald
```
### Output Distribution-Matching Regularization (ODMR)
Two plots named `odmr_max-entropy.pdf` and `odmr_bald.pdf` will be saved to `../figures/intent_classification/` directory.
```bash
python odmr.py --criterion max-entropy
python odmr.py --criterion bald
```

## Compile and print optimal, heuristic, and random quality numbers for LSTM
```bash
python collect_results.py
```
