
import argparse, random, time, string, pickle, os, nltk
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from utils import load_optimal, load_baseline
import help_text as ht

def plot_ref_meter(ref_cdf):
    for i in range(len(ref_cdf) - 1):
        rect = patches.Rectangle((0, ref_cdf[i]), 10, ref_cdf[i+1]-ref_cdf[i],
            edgecolor='none', facecolor=f'C{i}', zorder=-1000, alpha=0.7)
        plt.gca().add_patch(rect)
    plt.axis([0, 10, -0.05, 1.05])
    plt.axis('off')

def group_proportions(props, max_prop):
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

def get_len_proportion(order, warmstart_sents, pool_sents, len_groups, tot_acq):
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
    for o in order[:tot_acq]:
        sent = pool_sents[o]
        lg_counts.update([get_len_group(len(sent))])
        prop_trajs.append(get_proportion())
    prop_trajs = np.array(prop_trajs).T
    for i in range(1, num_groups):
        prop_trajs[i] += prop_trajs[i-1]
    prop_trajs = list(prop_trajs)
    prop_trajs.insert(0, [0] * len(prop_trajs[0]))
    return prop_trajs

def plot_len_proportion(order, warmstart_sents, pool_sents, len_groups, len_ref, tot_acq):
    N_warmstart = len(warmstart_sents)
    xs = range(N_warmstart, N_warmstart + tot_acq + 1)
    prop_trajs = get_len_proportion(order, warmstart_sents, pool_sents, len_groups, tot_acq)
    for i in range(1, len(len_groups)):
        plt.plot(xs, [len_ref[i]] * len(xs), 'k', dashes=[10, 5], lw=0.5)
    for i, (t1, t2) in enumerate(zip(prop_trajs[0:], prop_trajs[1:])):
        plt.fill_between(xs, t1, t2, facecolor=f'C{i}', edgecolor=f'C{i}', lw=0., alpha=0.7)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([N_warmstart, N_warmstart + tot_acq, -0.05, 1.05])

def get_tag_proportion(order, warmstart_tags, pool_tags, tag_groups_dict, tot_acq):
    def flatten_and_map(tags):
        return [tag_groups_dict[t] for t in sum(map(list, tags), [])]
    num_groups = len(set(tag_groups_dict.values()))
    tag_counts = Counter()
    tag_counts.update(flatten_and_map(warmstart_tags))
    prop_trajs = []
    counts = np.array([tag_counts[i] for i in range(num_groups)])
    prop_trajs.append(counts / counts.sum())
    for o in order[:tot_acq]:
        cur_tags = [pool_tags[o]]
        tag_counts.update(flatten_and_map(cur_tags))
        counts = np.array([tag_counts[i] for i in range(num_groups)])
        prop_trajs.append(counts / counts.sum())
    prop_trajs = np.array(prop_trajs).T
    for i in range(1, num_groups):
        prop_trajs[i] += prop_trajs[i-1]
    prop_trajs = list(prop_trajs)
    prop_trajs.insert(0, [0] * len(prop_trajs[0]))
    return prop_trajs

def plot_tag_proportion(order, warmstart_tags, pool_tags, tag_ref, tag_groups_dict, tot_acq):
    N_warmstart = len(warmstart_tags)
    xs = range(N_warmstart, N_warmstart + tot_acq + 1)
    prop_trajs = get_tag_proportion(order, warmstart_tags, pool_tags, tag_groups_dict, tot_acq)
    for i in range(1, len(tag_ref)):
        plt.plot(xs, [tag_ref[i]] * len(xs), 'k', dashes=[10, 5], lw=0.5)
    for i, (t1, t2) in enumerate(zip(prop_trajs[0:], prop_trajs[1:])):
        plt.fill_between(xs, t1, t2, facecolor=f'C{i}', edgecolor=f'C{i}', lw=0., alpha=0.7)
    plt.axis([N_warmstart, N_warmstart + tot_acq, -0.05, 1.05])

def main(criterion, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq, log_dir):
    data = pickle.load(open('data/restaurant.pkl', 'rb'))['seeds'][data_seed]
    N_warmstart = len(data['warmstart'])
    N_pool = len(data['pool'])
    warmstart_sents, warmstart_tags = zip(*data['warmstart'])
    pool_sents, pool_tags = zip(*data['pool'])
    test_sents, test_tags = zip(*data['test'])
    display_name = {'min-confidence': 'Min-Confidence', 'normalized-min-confidence': 'Norm.-Min-Confidence',
                    'longest': 'Longest'}[criterion]

    optimal_order, _, _ = load_optimal(log_dir, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)
    heuristic_order = load_baseline(criterion, 'test', model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)['order']
    random.seed(0)
    random_order = random.sample(range(N_pool), N_pool)[:tot_acq]

    lens_ct = Counter([len(t) for t in test_sents])
    max_len = max(lens_ct.keys())
    lens_props = [lens_ct[i] / sum(lens_ct.values()) for i in range(max_len + 1)]
    len_groups = group_proportions(lens_props, 0.13)
    num_groups = len(len_groups)
    group_sizes = np.array([sum([lens_ct[l] for l in len_group]) for len_group in len_groups])
    len_ref = np.cumsum(group_sizes / sum(group_sizes))
    len_ref = list(len_ref.flat)
    len_ref.insert(0, 0)

    tag_groups = [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    tag_groups_dict = {v: i for i, vs in enumerate(tag_groups) for v in vs}
    _, test_tags = zip(*data['test'])
    test_tags = [tag_groups_dict[t] for t in sum(map(list, test_tags), [])]
    test_tag_counts = Counter(test_tags)
    test_tag_counts = np.array([test_tag_counts[i] for i in range(len(tag_groups))])
    tag_ref = np.cumsum(test_tag_counts / sum(test_tag_counts))
    tag_ref = list(tag_ref.flat)
    tag_ref.insert(0, 0)

    fig = plt.figure(figsize=[9, 6])
    gs = GridSpec(ncols=5, nrows=2, width_ratios=[10, 0.7, 10, 0.7, 10], wspace=0.05, hspace=0.05)
    xs = np.linspace(N_warmstart, N_warmstart + tot_acq, 6)

    plt.subplot(gs[0, 0])
    plot_len_proportion(optimal_order, warmstart_sents, pool_sents, len_groups, len_ref, tot_acq)
    plt.xticks([])
    plt.ylabel('Length Distribution')
    plt.title('Optimal Order')

    plt.subplot(gs[0, 1])
    plot_ref_meter(len_ref)

    plt.subplot(gs[0, 2])
    plot_len_proportion(heuristic_order, warmstart_sents, pool_sents, len_groups, len_ref, tot_acq)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{display_name} Order')

    plt.subplot(gs[0, 3])
    plot_ref_meter(len_ref)

    plt.subplot(gs[0, 4])
    plot_len_proportion(random_order, warmstart_sents, pool_sents, len_groups, len_ref, tot_acq)
    plt.xticks([])
    plt.yticks([])
    plt.title('Random Order')

    plt.subplot(gs[1, 0])
    plot_tag_proportion(optimal_order, warmstart_tags, pool_tags, tag_ref, tag_groups_dict, tot_acq)
    plt.ylabel('Tag Distribution')
    plt.xlabel('# Data Points')

    plt.subplot(gs[1, 1])
    plot_ref_meter(tag_ref)

    plt.subplot(gs[1, 2])
    plot_tag_proportion(heuristic_order, warmstart_tags, pool_tags, tag_ref, tag_groups_dict, tot_acq)
    plt.yticks([])
    plt.xlabel('# Data Points')

    plt.subplot(gs[1, 3])
    plot_ref_meter(tag_ref)

    plt.subplot(gs[1, 4])
    plot_tag_proportion(random_order, warmstart_tags, pool_tags, tag_ref, tag_groups_dict, tot_acq)
    plt.xticks(xs)
    plt.yticks([])
    plt.xlabel('# Data Points')

    plt.savefig(f'../figures/named_entity_recognition/distribution_vis_{criterion}.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser(description='Visualize acquired data distribution')
    ###################### required args ######################
    parser.add_argument('--criterion', type=str, help=ht.criterion)
    ###########################################################
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=250, help=ht.tot_acq)
    parser.add_argument('--log-dir', type=str, default='logs', help=ht.log_dir)
    args = parser.parse_args()
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
