
import argparse, os, shelve
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

from utils import load_optimal, load_baseline

def plot_relative_order(order1, order2, name1=None, name2=None):
    overlap = list(set(order1).intersection(set(order2)))
    N = len(order1)
    H = N * 3
    W = int(N * 3 / 8)
    arr = np.zeros((H, W)) + 0.6
    for i in range(N)[::2]:
        arr[i*3:i*3+3] = 0.4
    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(arr, cmap='Blues', vmin=0, vmax=1)
    if name1 is not None:
        ax1.set_xlabel(name1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(arr, cmap='Oranges', vmin=0, vmax=1)
    if name2 is not None:
        ax2.set_xlabel(name2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    plt.suptitle(f'# Shared Data Points: {len(overlap)}')
    plt.tight_layout()
    for o in overlap:
        idx1 = order1.index(o)
        idx2 = order2.index(o)
        con = ConnectionPatch(xyA=(W - 0.5, idx1 * 3 + 1), coordsA=ax1.transData,
                              xyB=(  - 0.5, idx2 * 3 + 1), coordsB=ax2.transData)
        fig.add_artist(con)

def main(model1='lstm', criterion1='bald', model2='cnn', criterion2='bald', model_seed=0, 
         domain='alarm', data_seed=0, batchsize=20, max_epoch=100, patience=20, tot_acq=160, log_dir='logs'):
    m_dict = {'lstm': 'LSTM', 'cnn': 'CNN', 'aoe': 'AOE', 'roberta': 'RoBERTa'}
    c_dict = {'max-entropy': 'Max-Entropy', 'bald': 'BALD'}
    optimal_order1, _, _ = load_optimal(log_dir, model1, model_seed,
                               domain, data_seed, batchsize, max_epoch, patience, tot_acq)
    optimal_order2, _, _ = load_optimal(log_dir, model2, model_seed,
                               domain, data_seed, batchsize, max_epoch, patience, tot_acq)
    plot_relative_order(optimal_order1, optimal_order2, 
                        f'{m_dict[model1]}\nOptimal', f'{m_dict[model2]}\nOptimal')
    plt.savefig(f'../figures/intent_classification/relative_orders/{model1}_{model2}_optimal.pdf', 
        bbox_inches='tight')
    
    heuristic_order1 = load_baseline(criterion1, 'test', model1, model_seed, 
                              domain, data_seed, batchsize, max_epoch, patience, tot_acq)['order']
    heuristic_order2 = load_baseline(criterion2, 'test', model2, model_seed, 
                               domain, data_seed, batchsize, max_epoch, patience, tot_acq)['order']
    plot_relative_order(heuristic_order1, heuristic_order2, 
                        f'{m_dict[model1]}\n{c_dict[criterion1]}', f'{m_dict[model2]}\n{c_dict[criterion2]}')
    plt.savefig(f'../figures/intent_classification/relative_orders/{model1}_{model2}_heuristic.pdf', 
        bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, default='lstm')
    parser.add_argument('--criterion1', type=str, default='bald')
    parser.add_argument('--model2', type=str, default='cnn')
    parser.add_argument('--criterion2', type=str, default='bald')
    parser.add_argument('--model-seed', type=int, default=0)

    parser.add_argument('--domain', type=str, default='alarm')
    parser.add_argument('--data-seed', type=str, default=0)
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--tot-acq', type=int, default=160)

    parser.add_argument('--log-dir', type=str, default='logs')
    args = parser.parse_args()

    main(**vars(args))

if __name__ == '__main__':
    main_cli()
