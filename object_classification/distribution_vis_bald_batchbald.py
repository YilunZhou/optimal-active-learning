
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
    print(np.linspace(len(warmstart_y), len(warmstart_y) + len(order), 5))
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

def main(model_seed, data_seed, batchsize, max_epoch, patience, tot_acq, log_dir):
    data = load_data(data_seed)
    warmstart = data['warmstart']
    pool = data['pool']
    test = data['test']

    optimal_order, _, _ = load_optimal(log_dir, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)
    bald_order = load_baseline('bald', 'test', model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)['order']
    batchbald_order = load_baseline('batchbald', 'test', model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)['order']
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

    plt.figure(figsize=[10, 7])
    gs = GridSpec(ncols=7, nrows=2,
                  width_ratios=[10, 0.7, 10, 0.7, 10, 0.7, 10], wspace=0.05,
                  height_ratios=[3, 2], hspace=0.15)

    plt.subplot(gs[0, :3])
    plot_tsne(bald_order, warmstart_X_2d, warmstart_y, pool_X_2d, pool_y, test_X_2d, test_y[:2000], 5, batchsize)
    plt.title('BALD Order')

    plt.subplot(gs[0, 4:])
    plot_tsne(batchbald_order, warmstart_X_2d, warmstart_y, pool_X_2d, pool_y, test_X_2d, test_y[:2000], 5, batchsize)
    plt.title('BatchBALD Order')

    plt.subplot(gs[1, 0])
    plot_label_proportion(optimal_order, warmstart_y, pool_y, test_y)
    plt.title('Optimal Order')
    plt.ylabel('Label Distribution')

    plt.subplot(gs[1, 1])
    plot_ref_meter(test_y)

    plt.subplot(gs[1, 2])
    plot_label_proportion(bald_order, warmstart_y, pool_y, test_y)
    plt.title(f'BALD Order')
    plt.yticks([])

    plt.subplot(gs[1, 3])
    plot_ref_meter(test_y)

    plt.subplot(gs[1, 4])
    plot_label_proportion(batchbald_order, warmstart_y, pool_y, test_y)
    plt.title(f'BatchBALD Order')
    plt.yticks([])

    plt.subplot(gs[1, 5])
    plot_ref_meter(test_y)

    plt.subplot(gs[1, 6])
    plot_label_proportion(random_order, warmstart_y, pool_y, test_y)
    plt.title('Random Order')
    plt.yticks([])

    plt.savefig(f'../figures/object_classification/distribution_vis_bald_batchbald.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser('Visualize acquired data distribution')
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
