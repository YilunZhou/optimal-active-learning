
import argparse, os, shelve
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from train_scheduler import TrainScheduler
from utils import load_optimal, load_baseline

def display_name(name):
    if name == 'roberta':
        return 'RoBERTa'
    else:
        return name.upper()

def compute_quality(criterion, search_model, search_seed, eval_model, eval_seed,
                 domain, data_seed, batchsize, max_epoch, patience, tot_acq,
                 use_gpus, workers_per_gpu, log_dir):
    if criterion == 'random':
        curve = load_baseline('random', 'test', eval_model, eval_seed,
                              domain, data_seed, batchsize, max_epoch, patience, tot_acq)['curve']
        return np.mean(curve)
    spec1 = f'{criterion} {search_model} {search_seed}'
    spec2 = f'{eval_model} {eval_seed}'
    spec3 = f'{domain} {data_seed} {batchsize} {max_epoch} {patience} {tot_acq}'
    spec = f'{spec1} | {spec2} | {spec3}'
    if criterion == 'optimal':
        order, _, _ = load_optimal(log_dir, search_model, search_seed,
                                   domain, data_seed, batchsize, max_epoch, patience, tot_acq)
    else:
        order = load_baseline(criterion, 'test', search_model, eval_seed,
                              domain, data_seed, batchsize, max_epoch, patience, tot_acq)['order']
    with shelve.open('statistics/model_transfer.shv') as transfer:
        if spec in transfer and transfer[spec]['order'] == order:  # fresh copy of cache
            return np.mean(transfer[spec]['curve'])
        # not present or stale copy of cache
        train_args = Namespace(model=eval_model, model_seed=eval_seed,
                               domain=domain, data_seed=data_seed, evaluation_set='test',
                               batchsize=batchsize, max_epoch=max_epoch, patience=patience,
                               tot_acq=tot_acq, use_gpus=use_gpus, workers_per_gpu=workers_per_gpu)
        scheduler = TrainScheduler(train_args)
        curve = scheduler.evaluate_order(order)
        transfer[spec] = {'curve': curve, 'order': order}
        return np.mean(curve)


def main(models=['lstm', 'cnn', 'aoe', 'roberta'], criterions=['bald', 'bald', 'max-entropy', 'bald'],
         model_seed=0, domain='alarm', data_seed=0, batchsize=20, max_epoch=100, patience=20, tot_acq=160,
         use_gpus='all', workers_per_gpu=1, log_dir='logs'):
    num_models = len(models)
    optimal_matrix = np.zeros((num_models, num_models))
    optimal_diff = np.zeros((num_models, num_models))
    heuristic_matrix = np.zeros((num_models, num_models))
    heuristic_diff = np.zeros((num_models, num_models))
    random_matrix = np.zeros((num_models, 1))
    random_diff = np.zeros((num_models, 1))

    for i, eval_model in enumerate(models):
        random_quality = compute_quality('random', None, None, eval_model, model_seed,
                                        domain, data_seed, batchsize, max_epoch, patience, tot_acq,
                                        None, None, None)
        random_matrix[i, 0] = random_quality
        for j, (search_model, criterion) in enumerate(zip(models, criterions)):
            optimal_quality = compute_quality('optimal', search_model, model_seed, eval_model, model_seed,
                                              domain, data_seed, batchsize, max_epoch, patience, tot_acq,
                                              use_gpus, workers_per_gpu, log_dir)
            heuristic_quality = compute_quality(criterion, search_model, model_seed, eval_model, model_seed,
                                                domain, data_seed, batchsize, max_epoch, patience, tot_acq,
                                                use_gpus, workers_per_gpu, log_dir)
            optimal_matrix[i, j] = optimal_quality
            optimal_diff[i, j] = optimal_quality - random_quality
            heuristic_matrix[i, j] = heuristic_quality
            heuristic_diff[i, j] = heuristic_quality - random_quality

    plt.figure(figsize=[9, 3.5])
    gs = GridSpec(nrows=1, ncols=4, width_ratios=[4, 1, 4, 0.2])
    color_range = max(abs(optimal_diff).max(), abs(heuristic_diff).max())

    plt.subplot(gs[0, 0])
    plt.imshow(optimal_diff, cmap='coolwarm', vmin=-color_range, vmax=color_range, aspect='auto')
    for i in range(num_models):
        for j in range(num_models):
            plt.annotate(f'{optimal_matrix[i, j]:0.3f}', xy=(j, i), ha='center', va='center', fontsize=13)
            rect = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, linewidth=1)
            plt.gca().add_patch(rect)
    plt.xticks(range(num_models), [display_name(m) for m in models])
    plt.xlabel('Source Architecture')
    plt.yticks(range(num_models), [display_name(m) for m in models], rotation=90, va='center')
    plt.ylabel('Target Architecture')
    plt.title('Optimal')

    plt.subplot(gs[0, 1])
    plt.imshow(random_diff, cmap='coolwarm', vmin=-color_range, vmax=color_range, aspect='auto')
    for i in range(num_models):
            plt.annotate(f'{random_matrix[i, 0]:0.3f}', xy=(0, i), ha='center', va='center', fontsize=13)
            rect = plt.Rectangle((-0.5, i - 0.5), 1, 1, fill=False, linewidth=1)
            plt.gca().add_patch(rect)
    plt.xticks([])
    plt.yticks([])
    plt.title('Random')

    plt.subplot(gs[0, 3])  # plot the colorbar in the last subplot.
    cax = plt.gca()

    plt.subplot(gs[0, 2])
    plt.imshow(heuristic_diff, cmap='coolwarm', vmin=-color_range, vmax=color_range, aspect='auto')
    for i in range(num_models):
        for j in range(num_models):
            plt.annotate(f'{heuristic_matrix[i, j]:0.3f}', xy=(j, i), ha='center', va='center', fontsize=13)
            rect = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, linewidth=1)
            plt.gca().add_patch(rect)
    plt.xticks(range(num_models), [display_name(m) for m in models])
    plt.xlabel('Source Achitecture')
    plt.yticks([])
    plt.title('Heuristic')
    plt.colorbar(cax=cax)

    plt.savefig('../figures/intent_classification/model_transfer.pdf', bbox_inches='tight')


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', default=['lstm', 'cnn', 'aoe', 'roberta'])
    parser.add_argument('--criterions', type=str, nargs='+', default=['bald', 'bald', 'max-entropy', 'bald'])
    parser.add_argument('--model-seed', type=int, default=0)

    parser.add_argument('--domain', type=str, default='alarm')
    parser.add_argument('--data-seed', type=str, default=0)
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--tot-acq', type=int, default=160)

    parser.add_argument('--use-gpus', type=str, default='all')
    parser.add_argument('--workers-per-gpu', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default='logs')
    args = parser.parse_args()

    main(**vars(args))

if __name__ == '__main__':
    main_cli()
