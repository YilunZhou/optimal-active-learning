
import argparse, random, time, string, pickle, os, nltk
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from utils import load_optimal, load_baseline

def get_label_cdf(order, warmstart_labels, pool_labels, num_labels):
    label_counts = Counter()
    label_counts.update(warmstart_labels)
    prop_trajs = []
    counts = np.array([label_counts[i] for i in range(num_labels)])
    prop_trajs.append(counts / counts.sum())
    for o in order:
        label_counts.update([pool_labels[o]])
        counts = np.array([label_counts[i] for i in range(num_labels)])
        prop_trajs.append(counts / counts.sum())
    prop_trajs = np.array(prop_trajs).T
    for i in range(1, num_labels):
        prop_trajs[i] += prop_trajs[i-1]
    prop_trajs = list(prop_trajs)
    prop_trajs.insert(0, [0] * len(prop_trajs[0]))
    return prop_trajs

def group_adjacent(props, max_prop):
    groups = []
    cur_group = []
    cur_idxs = []
    for i, p in enumerate(props):
        cur_group.append(p)
        cur_idxs.append(i)
        if sum(cur_group) >= max_prop:
            del cur_group[-1], cur_idxs[-1]
            groups.append(cur_idxs)
            cur_group = [p]
            cur_idxs = [i]
    groups.append(cur_idxs)
    return groups

def get_length_cdf(order, warmstart_sents, pool_sents, len_groups):
    num_groups = len(len_groups)
    len_group_dict = dict()
    for g, lg in enumerate(len_groups):
        for l in lg:
            len_group_dict[l] = g
    def get_len_group(le):
        if le in len_group_dict:
            return len_group_dict[le]
        else:  # last len_group
            return len(len_groups) - 1
    prop_trajs = []
    lg_counts = Counter()
    def get_proportion():
        counts = np.array([lg_counts[i] for i in range(num_groups)])
        return counts / counts.sum()
    lg_counts.update([get_len_group(len(sent)) for sent in warmstart_sents])
    prop_trajs.append(get_proportion())
    for o in order:
        sent = pool_sents[o]
        lg_counts.update([get_len_group(len(sent))])
        prop_trajs.append(get_proportion())
    prop_trajs = np.array(prop_trajs).T
    for i in range(1, num_groups):
        prop_trajs[i] += prop_trajs[i-1]
    prop_trajs = list(prop_trajs)
    prop_trajs.insert(0, [0] * len(prop_trajs[0]))
    return prop_trajs

def plot_length_distribution(order, warmstart_sents, pool_sents, len_groups, ref_cdf, xs):
    N_warmstart = len(warmstart_sents)
    tot_acq = len(order)
    prop_trajs = get_length_cdf(order, warmstart_sents, pool_sents, len_groups)
    for i, (t1, t2) in enumerate(zip(prop_trajs[:-1], prop_trajs[1:])):
        plt.fill_between(xs, t1, t2, facecolor=f'C{i}', edgecolor='none', alpha=0.7)
    plt.axis([N_warmstart, N_warmstart + tot_acq, -0.05, 1.05])
    for i in range(0, len(len_groups)):
        if i > 0:
            plt.plot(xs, [ref_cdf[i]] * len(xs), 'k', dashes=[10, 5], lw=0.5)

def plot_label_distribution(order, warmstart_labels, pool_labels, num_labels, ref_cdf, xs):
    N_warmstart = len(warmstart_labels)
    tot_acq = len(order)
    prop_trajs = get_label_cdf(order, warmstart_labels, pool_labels, num_labels)
    plt.axis([N_warmstart, N_warmstart + tot_acq, -0.05, 1.05])
    for i in range(0, num_labels):
        if i > 0:
            plt.plot(xs, [ref_cdf[i]] * len(xs), 'k', dashes=[10, 5], lw=0.5)
    for i, (t1, t2) in enumerate(zip(prop_trajs[:-1], prop_trajs[1:])):
        plt.fill_between(x=xs, y1=t1, y2=t2, facecolor=f'C{i}', edgecolor=f'C{i}', lw=0., alpha=0.7)

def plot_ref_meter(ref_cdf):
    for i in range(len(ref_cdf) - 1):
        rect = patches.Rectangle((0, ref_cdf[i]), 10, ref_cdf[i+1]-ref_cdf[i],
            edgecolor='none', facecolor=f'C{i}', zorder=-1000, alpha=0.7)
        plt.gca().add_patch(rect)
    plt.axis([0, 10, -0.05, 1.05])
    plt.axis('off')

def main(model='lstm', model_seed=0, domain='alarm', data_seed=0,
         batchsize=20, max_epoch=100, patience=20, tot_acq=160, log_dir='logs'):
    data = pickle.load(open('data/TOP.pkl', 'rb'))[domain]
    num_labels = int(len(data['intent_label_mapping']) / 2)
    data = data['seeds'][data_seed]
    N = len(data['pool'])
    N_warmstart = len(data['warmstart'])

    optimal_order, _, _ = load_optimal(log_dir, model, model_seed, domain, data_seed, batchsize, max_epoch, patience, tot_acq)
    criterions = [('max-entropy', 'Max-Entropy'), ('bald', 'BALD')]
    heuristic_orders = [load_baseline(c, 'test', model, model_seed, domain, data_seed,
                                      batchsize, max_epoch, patience, tot_acq)['order'] for c, _ in criterions]
    random.seed(0)
    random_order = random.sample(range(N), N)[:tot_acq]

    warmstart_sents, warmstart_labels = zip(*data['warmstart'])
    warmstart_sents = [nltk.word_tokenize(sent) for sent in warmstart_sents]
    pool_sents, pool_labels = zip(*data['pool'])
    pool_sents = [nltk.word_tokenize(sent) for sent in pool_sents]
    test_sents, test_labels = zip(*data['test'])
    test_sents = [nltk.word_tokenize(sent) for sent in test_sents]
    lens_ct = Counter([len(t) for t in test_sents])
    max_len = max(lens_ct.keys())
    lens_props = [lens_ct[i] / sum(lens_ct.values()) for i in range(max_len+1)]
    len_groups = group_adjacent(lens_props, 0.08)
    print('Length grouping:', len_groups)

    group_sizes = np.array([sum([lens_ct[l] for l in len_group]) for len_group in len_groups])
    len_ref_cdf = np.cumsum(group_sizes / sum(group_sizes))
    len_ref_cdf = list(len_ref_cdf.flat)
    len_ref_cdf.insert(0, 0)

    test_label_counts = Counter(test_labels)
    test_label_counts = np.array([test_label_counts[i] for i in range(num_labels)])
    label_ref_cdf = np.cumsum(test_label_counts / sum(test_label_counts))
    label_ref_cdf = list(label_ref_cdf.flat)
    label_ref_cdf.insert(0, 0)

    fig = plt.figure(figsize=[12, 5.5])
    gs = GridSpec(ncols=7, nrows=2, width_ratios=[10, 0.7, 10, 0.7, 10, 0.7, 10], wspace=0.05, hspace=0.05)
    xs = range(N_warmstart, tot_acq + N_warmstart + 1)

    fig.add_subplot(gs[0, 0])
    plot_length_distribution(optimal_order, warmstart_sents, pool_sents, len_groups, len_ref_cdf, xs)
    plt.title('Optimal')
    plt.xticks([])
    plt.ylabel('Sentence Length Distribution')

    fig.add_subplot(gs[0, 1])
    plot_ref_meter(len_ref_cdf)

    fig.add_subplot(gs[0, 2])
    plot_length_distribution(heuristic_orders[0], warmstart_sents, pool_sents, len_groups, len_ref_cdf, xs)
    plt.title(f'{criterions[0][1]}')
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(gs[0, 3])
    plot_ref_meter(len_ref_cdf)

    fig.add_subplot(gs[0, 4])
    plot_length_distribution(heuristic_orders[1], warmstart_sents, pool_sents, len_groups, len_ref_cdf, xs)
    plt.title(f'{criterions[1][1]}')
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(gs[0, 5])
    plot_ref_meter(len_ref_cdf)

    fig.add_subplot(gs[0, 6])
    plot_length_distribution(random_order, warmstart_sents, pool_sents, len_groups, len_ref_cdf, xs)
    plt.title('Random')
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(gs[1, 0])
    plot_label_distribution(optimal_order, warmstart_labels, pool_labels, num_labels, label_ref_cdf, xs)
    plt.xlabel('# Data Points')
    plt.xticks(np.linspace(N_warmstart, N_warmstart + tot_acq, 5))
    plt.ylabel('Label Distribution')

    fig.add_subplot(gs[1, 1])
    plot_ref_meter(label_ref_cdf)

    fig.add_subplot(gs[1, 2])
    plot_label_distribution(heuristic_orders[0], warmstart_labels, pool_labels, num_labels, label_ref_cdf, xs)
    plt.xlabel('# Data Points')
    plt.xticks(np.linspace(N_warmstart, N_warmstart + tot_acq, 5))
    plt.yticks([])

    fig.add_subplot(gs[1, 3])
    plot_ref_meter(label_ref_cdf)

    fig.add_subplot(gs[1, 4])
    plot_label_distribution(heuristic_orders[1], warmstart_labels, pool_labels, num_labels, label_ref_cdf, xs)
    plt.xlabel('# Data Points')
    plt.xticks(np.linspace(N_warmstart, N_warmstart + tot_acq, 5))
    plt.yticks([])

    fig.add_subplot(gs[1, 5])
    plot_ref_meter(label_ref_cdf)

    fig.add_subplot(gs[1, 6])
    plot_label_distribution(random_order, warmstart_labels, pool_labels, num_labels, label_ref_cdf, xs)
    plt.xlabel('# Data Points')
    plt.xticks(np.linspace(N_warmstart, N_warmstart + tot_acq, 5))
    plt.yticks([])

    plt.savefig('../figures/intent_classification/distribution_vis.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser(description='Analyze result')
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--model-seed', type=int, default=0)
    parser.add_argument('--domain', type=str, default='alarm')
    parser.add_argument('--data-seed', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--tot-acq', type=int, default=160)
    parser.add_argument('--log-dir', type=str, default='logs')
    args = parser.parse_args()
    print(args)
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
