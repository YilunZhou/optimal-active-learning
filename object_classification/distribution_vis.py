
import argparse, sys, os, math, pickle, random
from copy import deepcopy as copy
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import load_optimal, load_baseline, load_data
import help_text as ht

def plot_tsne(order, warmstart_X, warmstart_y, pool_X, pool_y, test_X, test_y, num_batches, batchsize):
    plt.scatter(test_X[:, 0], test_X[:, 1], c=[f'C{i}' for i in test_y], alpha=0.05, linewidths=0)
    plt.scatter(warmstart_X[:, 0], warmstart_X[:, 1], c=[f'C{i}' for i in warmstart_y], marker=(3, 1, 0), s=64, alpha=0.4)
    for i in range(num_batches):
        use_order = order[i * batchsize : (i + 1) * batchsize]
        X = np.array([pool_X[o] for o in use_order])
        y = [pool_y[o] for o in use_order]
        plt.scatter(X[:, 0], X[:, 1], c=[f'C{i}' for i in y], marker=f'${i+1}$', s=64)
    plt.xticks([])
    plt.yticks([])

def get_frac(ys, labels):
    ct = Counter(ys)
    return [ct[l] / len(ys) for l in labels]

def plot_label_proportion(order, warmstart_y, pool_y, test_y):
    labels = list(range(10))
    test_frac = get_frac(test_y, labels)
    label_ref_cdf = list(np.cumsum(test_frac).flat)
    label_ref_cdf.insert(0, 0)
    acquired_fracs = []
    if isinstance(warmstart_y, np.ndarray):
        warstart_y = list(warmstart_y.flat)
    use_ys = list(copy(warmstart_y))
    acquired_fracs.append(get_frac(use_ys, labels))
    for o in order:
        use_ys.append(pool_y[o])
        acquired_fracs.append(get_frac(use_ys, labels))
    trajs = list(zip(*acquired_fracs))
    xs = range(len(warmstart_y), len(warmstart_y) + len(order) + 1)
    tot_traj = np.zeros(len(trajs[0]))
    for i, traj in enumerate(trajs):
        plt.fill_between(xs, tot_traj, traj + tot_traj, facecolor=f'C{i}', alpha=0.7, edgecolor='none')
        tot_traj += traj
    plt.axis([min(xs), max(xs), -0.05, 1.05])
    for i in range(0, len(labels)):
        if i > 0:
            plt.plot(xs, [label_ref_cdf[i]] * len(xs), 'k', dashes=[10, 5], lw=0.5)
    plt.xlabel('# Data Points')
    plt.xticks(np.linspace(len(warmstart_y), len(warmstart_y) + len(order), 5))

def plot_ref_meter(test_y):
    labels = list(range(10))
    test_frac = get_frac(test_y, labels)
    label_ref_cdf = list(np.cumsum(test_frac).flat)
    label_ref_cdf.insert(0, 0)
    for i in range(0, len(labels)):
        rect = patches.Rectangle((0, label_ref_cdf[i]), 10, label_ref_cdf[i+1]-label_ref_cdf[i],
            edgecolor='none', facecolor=f'C{i}', zorder=-1000, alpha=0.7)
        plt.gca().add_patch(rect)
    plt.axis([0, 10, -0.05, 1.05])
    plt.axis('off')

def main(criterion, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq, log_dir):
    data = load_data(data_seed)
    warmstart = data['warmstart']
    pool = data['pool']
    test = data['test']
    display_name = {'max-entropy': 'Max-Entropy', 'bald': 'BALD', 'batchbald': 'BatchBALD'}[criterion]

    optimal_order, _, _ = load_optimal(log_dir, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)
    heuristic_order = load_baseline(criterion, 'test', model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)['order']
    random.seed(0)
    random_order = random.sample(range(len(pool)), len(pool))[:tot_acq]

    warmstart_X, warmstart_y = zip(*warmstart)
    warmstart_X = np.array(warmstart_X)
    pool_X, pool_y = zip(*pool)
    pool_X = np.array(pool_X)
    test_X, test_y = zip(*test)
    test_X = np.array(test_X)

    all_X = np.vstack((warmstart_X, pool_X, test_X[:2000]))
    all_X = all_X.reshape(len(all_X), -1)
    pca = PCA(n_components=100, random_state=np.random.RandomState(0))
    tsne = TSNE(n_components=2, n_jobs=-1, random_state=np.random.RandomState(0))
    all_X_2d = tsne.fit_transform(pca.fit_transform(all_X))
    warmstart_X_2d, pool_X_2d, test_X_2d = np.array_split(all_X_2d, [len(warmstart_X), len(warmstart_X) + len(pool_X)])

    fig = plt.figure(figsize=[9, 7])
    gs1 = GridSpec(ncols=2, nrows=1, width_ratios=[1, 1], wspace=0.1, bottom=0.45)
    gs2 = GridSpec(ncols=5, nrows=1, width_ratios=[10, 0.7, 10, 0.7, 10], wspace=0.05, top=0.40)

    plt.subplot(gs1[0, 0])
    plot_tsne(optimal_order, warmstart_X_2d, warmstart_y, pool_X_2d, pool_y, test_X_2d, test_y[:2000], 5, batchsize)
    plt.title('Optimal Order')

    plt.subplot(gs1[0, 1])
    plot_tsne(heuristic_order, warmstart_X_2d, warmstart_y, pool_X_2d, pool_y, test_X_2d, test_y[:2000], 5, batchsize)
    plt.title(f'{display_name} Order')

    plt.subplot(gs2[0, 0])
    plot_label_proportion(optimal_order, warmstart_y, pool_y, test_y)
    plt.title('Optimal Order')
    plt.ylabel('Label Distribution')

    plt.subplot(gs2[0, 1])
    plot_ref_meter(test_y)

    plt.subplot(gs2[0, 2])
    plot_label_proportion(heuristic_order, warmstart_y, pool_y, test_y)
    plt.title(f'{display_name} Order')
    plt.yticks([])

    plt.subplot(gs2[0, 3])
    plot_ref_meter(test_y)

    plt.subplot(gs2[0, 4])
    plot_label_proportion(random_order, warmstart_y, pool_y, test_y)
    plt.title('Random Order')
    plt.yticks([])

    plt.savefig(f'../figures/object_classification/distribution_vis_{criterion}.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser('Visualize acquired data distribution')
    ###################### required args ######################
    parser.add_argument('--criterion', type=str, help=ht.criterion)
    ###########################################################
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=300, help=ht.tot_acq)
    parser.add_argument('--log-dir', type=str, default='logs', help=ht.log_dir)
    args = parser.parse_args()
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
