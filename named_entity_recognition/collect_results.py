
import argparse, shelve
import numpy as np
from utils import load_baseline
import help_text as ht

def main(model_seed=0, data_seed=0, batchsize=25, max_epoch=100, patience=20,
         tot_acq=250, criterions=['min-confidence', 'normalized-min-confidence', 'longest']):
    with shelve.open('statistics/perf_curves.shv') as curves:
        spec = f'{model_seed} {model_seed} {data_seed} {batchsize} {max_epoch} {patience} {tot_acq}'
        curve = curves[spec]['test']
        print(f'optimal quality: {np.mean(curve)}')

    for criterion in criterions:
        curve = load_baseline(criterion, 'test', model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)['curve']
        print(f'{criterion} quality: {np.mean(curve)}')

    curve = load_baseline('random', 'test', model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)['curve']
    print(f'Random AUC: {np.mean(curve)}')

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=250, help=ht.tot_acq)
    parser.add_argument('--criterions', type=str, nargs='+',
                        default=['min-confidence', 'normalized-min-confidence', 'longest'], help=ht.criterions)
    args = parser.parse_args()
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
