
import os, pickle, time, argparse, shelve
from copy import deepcopy as copy
from argparse import Namespace
from collections import Counter
from tqdm import trange

import numpy as np
import nltk
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.stats import entropy

import help_text as ht
from trainer import get_trainer
from utils import load_optimal, load_baseline, store_baseline
from heuristics import score_bald, score_maxent
from performance_curve import plot_curves
from distribution_vis import plot_label_distribution, plot_ref_meter, get_label_cdf

def prioritize_labels(cur_labels, label_ref):
    num_labels = len(label_ref)
    N = len(cur_labels)
    expected = [p * N for p in label_ref]
    actual_ct = Counter(cur_labels)
    actual = np.array([actual_ct[i] for i in range(num_labels)])
    deficits = actual - expected
    assert min(deficits) <= 0
    labels = np.argsort(deficits)
    return labels

def select_batch(idxs, use_labels, true_labels, train_labels, label_ref, batchsize):
    true_labels = {i: tl for i, tl in zip(idxs, true_labels)}
    use_idxs = []
    cur_labels = copy(list(train_labels))
    for _ in range(batchsize):
        target_labels = prioritize_labels(cur_labels, label_ref)
        for tgt_l in target_labels:
            found = False
            for i, l, tl in zip(idxs, use_labels, true_labels):
                if l == tgt_l and i not in use_idxs:
                    use_idxs.append(i)
                    cur_labels.append(true_labels[i])
                    found = True
                    break
            if found:
                break
    assert len(set(use_idxs)) == len(use_idxs) == batchsize
    return use_idxs

def odmr(data, criterion, evaluation_set, label_ref, imputed, model, model_seed, domain,
         batchsize, max_epoch, patience, tot_acq, gpu_idx, num_labels, smoothing=1):
    '''
    label_ref can be 'accessible' (which uses the warmstart + model_sel set to estimate label proportion) or
    'test' (which uses the test set to estimate label proportion).
    imputed can be True (which uses groundtruth label for balancing) or False (which uses predicted label)
    there are four combinations:
    1. label_ref = 'test', imputed = True
    2. label_ref = 'accessible', imputed = True
    3. label_ref = 'test', imputed = False
    4. label_ref = 'accessible', imputed = False <-- this is the only realistic setting
    during each batch-acquisition, all previously acquired labels are known, and used for deficit calculation.
    In addition, as soon as one data point is queried, the label is known and used for deficit calculation.
    In other words, the labeling procedure is not "batched".
    '''
    assert label_ref in ['accessible', 'test']
    assert imputed in [True, False]
    if label_ref == 'accessible':
        _, warmstart_labels = zip(*data['warmstart'])
        _, model_sel_labels = zip(*data['train_valid'])
        labels = list(warmstart_labels) + list(model_sel_labels)
    else:
        _, labels = zip(*data[evaluation_set])
    label_ct = Counter(labels)
    label_ref = np.array([label_ct[i] + smoothing for i in range(num_labels)]) / (len(labels) + smoothing * num_labels)

    train_set = copy(data['warmstart'])
    pool_dict = {i: p for i, p in enumerate(data['pool'])}
    model_sel_set = data['train_valid']
    eval_set = data[evaluation_set]
    f1s = []
    mcdropout = (criterion == 'bald')

    trainer = get_trainer(model, model_seed, domain, f'cuda:{gpu_idx}', mcdropout)
    trainer.train(train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False)
    f1 = trainer.evaluate_f1(trainer.best_model, eval_set)
    f1s.append(f1)
    data_order = []
    for _ in trange(int(tot_acq / batchsize)):
        pool_idxs = list(pool_dict.keys())
        pool_sents, pool_labels = zip(*pool_dict.values())
        if criterion == 'bald':
            pool_scores = score_bald(trainer.best_model, pool_sents)
        elif criterion == 'max-entropy':
            pool_scores = score_maxent(trainer.best_model, pool_sents)
        if imputed:
            use_labels = pool_labels
        else:
            with torch.no_grad():
                use_labels = trainer.best_model(pool_sents, evall=True).cpu().numpy().argmax(axis=1)
        _, train_labels = zip(*train_set)
        _, pool_idxs, use_labels, true_labels = zip(*sorted(zip(pool_scores, pool_idxs, use_labels, pool_labels)))
        use_idxs = select_batch(pool_idxs, use_labels, true_labels, train_labels, label_ref, batchsize)

        train_set = train_set + [pool_dict[i] for i in use_idxs]
        data_order = data_order + use_idxs
        for idx in use_idxs:
            del pool_dict[idx]

        trainer = get_trainer(model, model_seed, domain, f'cuda:{gpu_idx}', mcdropout)
        trainer.train(train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False)
        f1 = trainer.evaluate_f1(trainer.best_model, eval_set)
        f1s.append(f1)
    return f1s, data_order

def main(criterion, model='lstm', model_seed=0, domain='alarm', data_seed=0,
         batchsize=20, max_epoch=100, patience=20, tot_acq=160, evaluation_set='test', gpu_idx=0, log_dir='logs'):
    data = pickle.load(open('data/TOP.pkl', 'rb'))[domain]
    num_labels = int(len(data['intent_label_mapping']) / 2)
    data = data['seeds'][data_seed]
    _, warmstart_labels = zip(*data['warmstart'])
    _, pool_labels = zip(*data['pool'])
    N_warmstart = len(data['warmstart'])
    N = len(data['pool'])
    try:
        l1 = load_baseline(f'odmr-l1-{criterion}', evaluation_set, model, model_seed, domain, data_seed,
                                   batchsize, max_epoch, patience, tot_acq)
        l1_curve, l1_order = l1['curve'], l1['order']
        print(l1_curve, np.mean(l1_curve))
        l2 = load_baseline(f'odmr-l2-{criterion}', evaluation_set, model, model_seed, domain, data_seed,
                                   batchsize, max_epoch, patience, tot_acq)
        l2_curve, l2_order = l2['curve'], l2['order']
        print(l2_curve, np.mean(l2_curve))
        l3 = load_baseline(f'odmr-l3-{criterion}', evaluation_set, model, model_seed, domain, data_seed,
                                   batchsize, max_epoch, patience, tot_acq)
        l3_curve, l3_order = l3['curve'], l3['order']
        print(l3_curve, np.mean(l3_curve))
        l4 = load_baseline(f'odmr-l4-{criterion}', evaluation_set, model, model_seed, domain, data_seed,
                                   batchsize, max_epoch, patience, tot_acq)
        l4_curve, l4_order = l4['curve'], l4['order']
        print(l4_curve, np.mean(l4_curve))
    except KeyError:
        l1_curve, l1_order = odmr(data, criterion, evaluation_set, 'test', True, model, model_seed, domain,
                                  batchsize, max_epoch, patience, tot_acq, gpu_idx, num_labels, smoothing=1)
        print(l1_curve, np.mean(l1_curve))
        store_baseline(l1_curve, l1_order, f'odmr-l1-{criterion}', evaluation_set, model, model_seed,
                       domain, data_seed, batchsize, max_epoch, patience, tot_acq)
        l2_curve, l2_order = odmr(data, criterion, evaluation_set, 'accessible', True, model, model_seed, domain,
                                  batchsize, max_epoch, patience, tot_acq, gpu_idx, num_labels, smoothing=1)
        print(l2_curve, np.mean(l2_curve))
        store_baseline(l2_curve, l2_order, f'odmr-l2-{criterion}', evaluation_set, model, model_seed,
                       domain, data_seed, batchsize, max_epoch, patience, tot_acq)
        l3_curve, l3_order = odmr(data, criterion, evaluation_set, 'test', False, model, model_seed, domain,
                                  batchsize, max_epoch, patience, tot_acq, gpu_idx, num_labels, smoothing=1)
        print(l3_curve, np.mean(l3_curve))
        store_baseline(l3_curve, l3_order, f'odmr-l3-{criterion}', evaluation_set, model, model_seed,
                       domain, data_seed, batchsize, max_epoch, patience, tot_acq)
        l4_curve, l4_order = odmr(data, criterion, evaluation_set, 'accessible', False, model, model_seed, domain,
                                  batchsize, max_epoch, patience, tot_acq, gpu_idx, num_labels, smoothing=1)
        print(l4_curve, np.mean(l4_curve))
        store_baseline(l4_curve, l4_order, f'odmr-l4-{criterion}', evaluation_set, model, model_seed,
                       domain, data_seed, batchsize, max_epoch, patience, tot_acq)

    plt.figure(figsize=[20, 4])
    gs = GridSpec(ncols=9, nrows=1, width_ratios=[10, 0.3, 10, 0.7, 10, 0.7, 10, 0.7, 10], wspace=0.05)
    plt.subplot(gs[0, 0])
    if criterion == 'max-entropy':
        baselines = [('max-entropy', 'Max-Entropy', 0),
                     ('odmr-l1-max-entropy', 'Test + True', 1),
                     ('odmr-l2-max-entropy', 'Acce + True', 2),
                     ('odmr-l3-max-entropy', 'Test + Pred', 6),
                     ('odmr-l4-max-entropy', 'Acce + Pred', 8),
                     ('random', 'Random', 4)]
    elif criterion == 'bald':
        baselines = [('bald', 'BALD', 0),
                     ('odmr-l1-bald', 'Test + True', 1),
                     ('odmr-l2-bald', 'Acce + True', 2),
                     ('odmr-l3-bald', 'Test + Pred', 6),
                     ('odmr-l4-bald', 'Acce + Pred', 8),
                     ('random', 'Random', 4)]
    xs = list(range(N_warmstart, N_warmstart + tot_acq + 1, batchsize))
    optimal_order, optimal_quality, _ = load_optimal(log_dir, model, model_seed, domain, data_seed,
                                                     batchsize, max_epoch, patience, tot_acq)
    plot_curves(optimal_order, xs, evaluation_set, model, model_seed, model_seed, domain, data_seed,
                batchsize, max_epoch, patience, tot_acq, None, None, baselines)
    plt.xlabel('# Data Points')
    plt.ylabel('F1')
    plt.title('Intent Classification')

    xs = list(range(N_warmstart, N_warmstart + tot_acq + 1))
    _, test_labels = zip(*data['test'])
    test_label_counts = Counter(test_labels)
    test_label_counts = np.array([test_label_counts[i] for i in range(num_labels)])
    label_ref_cdf = np.cumsum(test_label_counts / sum(test_label_counts))
    label_ref_cdf = list(label_ref_cdf.flat)
    label_ref_cdf.insert(0, 0)

    plt.subplot(gs[0, 2])
    plot_label_distribution(l1_order, warmstart_labels, pool_labels, num_labels, label_ref_cdf, xs)
    plt.title('Test + Groundtruth')
    plt.xlabel('# Data Points')
    plt.yticks([])

    plt.subplot(gs[0, 3])
    plot_ref_meter(label_ref_cdf)

    plt.subplot(gs[0, 4])
    plot_label_distribution(l2_order, warmstart_labels, pool_labels, num_labels, label_ref_cdf, xs)
    plt.title('Accessible + Groundtruth')
    plt.xlabel('# Data Points')
    plt.yticks([])

    plt.subplot(gs[0, 5])
    plot_ref_meter(label_ref_cdf)

    plt.subplot(gs[0, 6])
    plot_label_distribution(l3_order, warmstart_labels, pool_labels, num_labels, label_ref_cdf, xs)
    plt.title('Test + Predicted')
    plt.xlabel('# Data Points')
    plt.yticks([])

    plt.subplot(gs[0, 7])
    plot_ref_meter(label_ref_cdf)

    plt.subplot(gs[0, 8])
    plot_label_distribution(l4_order, warmstart_labels, pool_labels, num_labels, label_ref_cdf, xs)
    plt.title('Accessible + Predicted')
    plt.xlabel('# Data Points')
    plt.yticks([])

    plt.savefig(f'../figures/intent_classification/odmr_{criterion}.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser(description='Output Distribution-Matching Regularization')
    ################## required args ##################
    parser.add_argument('--criterion', type=str, choices=['max-entropy', 'bald'], help=ht.criterion)
    ###################################################
    parser.add_argument('--model', type=str, default='lstm', help=ht.model)
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--domain', type=str, default='alarm', help=ht.domain)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=20, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=160, help=ht.tot_acq)
    parser.add_argument('--evaluation-set', type=str, default='test', help=ht.evaluation_set)
    parser.add_argument('--gpu-idx', type=int, default=0, help=ht.gpu_idx)
    parser.add_argument('--log-dir', type=str, default='logs', help=ht.log_dir)
    args = parser.parse_args()
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
