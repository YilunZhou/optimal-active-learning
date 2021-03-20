
import argparse, shelve, os
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt

from train_scheduler import TrainScheduler
from utils import load_optimal, load_baseline
import help_text as ht


def abs_path(p):
    return os.path.join(os.path.dirname(__file__), p)

def plot_curves(order, xs, evaluation_set, search_model_seed, eval_model_seed, data_seed,
                batchsize, max_epoch, patience, tot_acq, use_gpus, workers_per_gpu, baselines):
    legends = []
    spec = f'{search_model_seed} {eval_model_seed} {data_seed} {batchsize} {max_epoch} {patience} {tot_acq}'
    with shelve.open(abs_path('statistics/perf_curves.shv')) as curves:
        if spec in curves and curves[spec]['order'] == order:
            optimal_curve = curves[spec][evaluation_set]
        else:
            eval_args = Namespace(model_seed=eval_model_seed, data_seed=data_seed, batchsize=batchsize,
                                  max_epoch=max_epoch, patience=patience, evaluation_set=evaluation_set,
                                  tot_acq=tot_acq, use_gpus=use_gpus, workers_per_gpu=workers_per_gpu)
            scheduler = TrainScheduler(eval_args)
            optimal_curve = scheduler.evaluate_order(order)
    plt.plot(xs, optimal_curve, 'C3-o')
    legends.append(f'Optimal: {np.mean(optimal_curve):0.3f}')
    for name, display_name, color in baselines:
        try:
            curve = load_baseline(name, evaluation_set, eval_model_seed, data_seed,
                                  batchsize, max_epoch, patience, tot_acq)['curve']
            plt.plot(xs, curve, f'C{color}-o')
            legends.append(f'{display_name}: {np.mean(curve):0.3f}')
        except:
            print(f'{display_name} not found')
    plt.legend(legends)
    return optimal_curve

def main(search_model_seed, eval_model_seed=None, data_seed=0, tot_acq=300, batchsize=25, max_epoch=100, patience=20,
         use_gpus='all', workers_per_gpu=2, log_dir='logs'):
    if eval_model_seed is None:
        eval_model_seed = search_model_seed

    N_warmstart = 50
    optimal_order, optimal_quality, _ = load_optimal(log_dir, search_model_seed, data_seed,
                                                     batchsize, max_epoch, patience, tot_acq)
    print(f'optimal quality in log: {optimal_quality}')

    plt.figure(figsize=[7.5, 4])
    xs = list(range(N_warmstart, N_warmstart + tot_acq + 1, batchsize))
    baselines = [('max-entropy', 'Max-Entropy', 0), ('bald', 'BALD', 1), ('batchbald', 'BatchBALD', 2), ('random', 'Random', 4)]
    plot_args = [search_model_seed, eval_model_seed, data_seed, batchsize, max_epoch, patience,
                 tot_acq, use_gpus, workers_per_gpu, baselines]

    plt.subplot(1, 2, 1)
    valid_curve = plot_curves(optimal_order, xs, 'valid', *plot_args)
    xmin1, xmax1, ymin1, ymax1 = plt.axis()
    plt.xticks(np.linspace(N_warmstart, tot_acq + N_warmstart, 5))
    plt.xlabel('# Data Points')
    plt.ylabel('Accuracy')
    plt.title('Validation Set $\\mathcal{D}^V$')
    ax1 = plt.gca()

    plt.subplot(1, 2, 2)
    test_curve = plot_curves(optimal_order, xs, 'test', *plot_args)
    xmin2, xmax2, ymin2, ymax2 = plt.axis()
    plt.yticks([])
    plt.xticks(np.linspace(N_warmstart, tot_acq + N_warmstart, 5))
    plt.xlabel('# Data Points')
    ax2 = plt.gca()
    plt.title('Test Set $\\mathcal{D}^T$')

    ax1.set_xlim(min(xmin1, xmin2), max(xmax1, xmax2))
    ax1.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
    ax2.set_xlim(min(xmin1, xmin2), max(xmax1, xmax2))
    ax2.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
    plt.tight_layout()
    fn = f'../figures/object_classification/perf_curves/s{search_model_seed}_e{eval_model_seed}.pdf'
    plt.savefig(abs_path(fn), bbox_inches='tight')

    print(f'Validation quality: {np.mean(valid_curve)}; Test quality: {np.mean(test_curve)}')

    spec = f'{search_model_seed} {eval_model_seed} {data_seed} {batchsize} {max_epoch} {patience} {tot_acq}'
    with shelve.open(abs_path('statistics/perf_curves.shv')) as curves:
        curves[spec] = {'valid': valid_curve, 'test': test_curve, 'order': optimal_order}

def main_cli():
    parser = argparse.ArgumentParser(description='Plot performance curve for the optimal order. ')
    #################### required args ####################
    parser.add_argument('--search-model-seed', type=int, help=ht.search_model_seed)
    #######################################################
    parser.add_argument('--eval-model-seed', type=int, help=ht.eval_model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--tot-acq', type=int, default=300, help=ht.tot_acq)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--use-gpus', type=str, default='all', help=ht.use_gpus)
    parser.add_argument('--workers-per-gpu', type=int, default=1, help=ht.workers_per_gpu)
    parser.add_argument('--log-dir', type=str, default='logs', help=ht.log_dir)
    args = parser.parse_args()
    print(args)
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
