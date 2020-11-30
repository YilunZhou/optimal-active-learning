
import argparse, shelve
import numpy as np
from utils import load_baseline
import help_text as ht

def main(model_seeds=[0, 1, 2, 3, 4], data_seed=0, batchsize=25, max_epoch=100, patience=20,
         tot_acq=300, criterions=['max-entropy', 'bald', 'batchbald']):
    optimal = []
    with shelve.open('statistics/perf_curves.shv') as curves:
        for seed in model_seeds:
            spec = f'{seed} {seed} {data_seed} {batchsize} {max_epoch} {patience} {tot_acq}'
            curve = curves[spec]['test']
            optimal.append(np.mean(curve))
    print(f'optimal quality: {np.mean(optimal)}')

    for criterion in criterions:
        heuristic = []
        for seed in model_seeds:
            curve = load_baseline(criterion, 'test', seed, data_seed, batchsize, max_epoch, patience, tot_acq)['curve']
            heuristic.append(np.mean(curve))
        print(f'{criterion} quality: {np.mean(heuristic)}')

    random = []
    for seed in model_seeds:
        curve = load_baseline('random', 'test', seed, data_seed, batchsize, max_epoch, patience, tot_acq)['curve']
        random.append(np.mean(curve))
    print(f'Random AUC: {np.mean(random)}')

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help=ht.model_seeds)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=300, help=ht.tot_acq)
    parser.add_argument('--criterions', type=str, nargs='+', default=['max-entropy', 'bald', 'batchbald'], help=ht.criterions)
    args = parser.parse_args()
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
