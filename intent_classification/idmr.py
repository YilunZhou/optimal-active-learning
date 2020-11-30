
import os, pickle, time, argparse
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
from scipy.stats import entropy

from trainer import get_trainer
from utils import load_optimal, load_baseline, store_baseline
from heuristics import score_bald, score_maxent
from performance_curve import plot_curves

import help_text as ht

def prioritize_len_groups(lengths, ref_prop, len_gidx_dict, len_groups):
    N = len(lengths)
    expected = [p * N for p in ref_prop]
    group_idxs = [len_gidx_dict[l] for l in lengths]
    group_idxs_ct = Counter(group_idxs)
    deficits = [group_idxs_ct[gidx] - e for gidx, e in enumerate(expected)]
    assert min(deficits) <= 0
    sorted_idxs = np.argsort(deficits)
    return [len_groups[g_idx] for g_idx in sorted_idxs]

def select_batch(batchsize, idxs, lens, train_lens, ref_prop, len_gidx_dict, len_groups):
    use_idxs = []
    cur_lens = copy(train_lens)
    for _ in range(batchsize):
        lohis = prioritize_len_groups(cur_lens, ref_prop, len_gidx_dict, len_groups)
        for lo, hi in lohis:  # this loop enables falling back to 2nd, 3rd, etc. length group choices
            found = False
            for (i, l) in zip(idxs, lens):
                if lo <= l <= hi and i not in use_idxs:
                    use_idxs.append(i)
                    cur_lens.append(l)
                    found = True
                    break
            if found:
                break
    assert len(set(use_idxs)) == len(use_idxs) == batchsize
    return use_idxs

def idmr(data, criterion, evaluation_set, lens_proportions, model, model_seed, domain, data_seed,
         batchsize, max_epoch, patience, tot_acq, gpu_idx):
    '''
    lens_proportions is a dictionary from (len_low, len_high), both inclusive, to proportions.
    the acquisition order will try to preserve this proportion in the warmstart + active-acquired set
    '''
    len_groups = list(lens_proportions.keys())
    proportions = list(lens_proportions.values())
    len_groups_dict = {v: i for i, vs in enumerate(len_groups) for v in range(vs[0], vs[1]+1)}

    train_set = copy(data['warmstart'])
    pool_set = copy(data['pool'])
    pool_dict = {i: p for i, p in enumerate(pool_set)}
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
        pool_sents, _ = zip(*pool_dict.values())
        if criterion == 'bald':
            pool_scores = score_bald(trainer.best_model, pool_sents)
        elif criterion == 'max-entropy':
            pool_scores = score_maxent(trainer.best_model, pool_sents)
        sorted_pool_scores, sorted_pool_idxs = zip(*sorted(zip(pool_scores, pool_idxs)))
        pool_lens = [len(nltk.word_tokenize(pool_dict[p][0])) for p in sorted_pool_idxs]
        train_lens = [len(nltk.word_tokenize(t[0])) for t in train_set]
        use_idxs = select_batch(batchsize, sorted_pool_idxs, pool_lens, train_lens, proportions, len_groups_dict, len_groups)
        train_set = train_set + [pool_dict[i] for i in use_idxs]
        data_order = data_order + use_idxs
        for idx in use_idxs:
            del pool_dict[idx]
        trainer = get_trainer(model, model_seed, domain, f'cuda:{gpu_idx}', mcdropout)
        trainer.train(train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False)
        f1 = trainer.evaluate_f1(trainer.best_model, eval_set)
        f1s.append(f1)
    return f1s, data_order

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

def main(criterion, model='lstm', model_seed=0, domain='alarm', data_seed=0,
         batchsize=20, max_epoch=100, patience=20, tot_acq=160, evaluation_set='test', gpu_idx=0, log_dir='logs'):
    assert model != 'roberta', 'IDMR for RoBERTa model is not implemented'
    data = pickle.load(open('data/TOP.pkl', 'rb'))[domain]['seeds'][data_seed]
    N_warmstart = len(data['warmstart'])
    N = len(data['pool'])
    try:
        idmr_curve = load_baseline(f'idmr-{criterion}', evaluation_set, model, model_seed, domain, data_seed,
                                   batchsize, max_epoch, patience, tot_acq)['curve']
    except KeyError:
        lens_proportions = None
        accessible_set = data['warmstart'] + data['train_valid'] + data['pool']
        accessible_sents, _ = zip(*accessible_set)
        lens = [len(nltk.word_tokenize(s)) for s in accessible_sents]
        lens_ct = Counter(lens)
        cts = np.array([lens_ct[l] for l in range(max(lens_ct.keys()) + 1)])
        props = cts / sum(cts)
        groups = group_proportions(props, 0.08)
        lens_proportions = dict()
        for i, g in enumerate(groups):
            lo, hi = min(g), max(g)
            sum_props = sum(props[p] for p in range(lo, hi + 1))
            if i == len(groups) - 1:
                hi = 100
            lens_proportions[(lo, hi)] = sum_props
        idmr_curve, idmr_order = idmr(data, criterion, evaluation_set, lens_proportions, model, model_seed,
                                      domain, data_seed, batchsize, max_epoch, patience, tot_acq, gpu_idx)
        print(idmr_curve)
        print(np.mean(idmr_curve))
        store_baseline(idmr_curve, idmr_order, f'idmr-{criterion}', evaluation_set, model, model_seed,
                       domain, data_seed, batchsize, max_epoch, patience, tot_acq)
    plt.figure()
    xs = list(range(N_warmstart, N_warmstart + tot_acq + 1, batchsize))
    baselines = [('max-entropy', 'Max-Entropy', 0), ('bald', 'BALD', 1), ('random', 'Random', 4)]
    if criterion == 'max-entropy':
        baselines.append(('idmr-max-entropy', 'IDMR Max-Ent.', 6))
    elif criterion == 'bald':
        baselines.append(('idmr-bald', 'IDMR BALD', 6))
    optimal_order, _, _ = load_optimal(log_dir, model, model_seed, domain, data_seed,
                                       batchsize, max_epoch, patience, tot_acq)
    plot_curves(optimal_order, xs, evaluation_set, model, model_seed, model_seed, domain, data_seed,
                batchsize, max_epoch, patience, tot_acq, None, None, baselines)
    xmin1, xmax1, ymin1, ymax1 = plt.axis()
    plt.xticks(np.linspace(N_warmstart, tot_acq + N_warmstart, 5))
    plt.xlabel('# Data Points')
    plt.ylabel('F1')
    plt.title('Input Distribution-Matching Regularization')
    plt.savefig(f'../figures/intent_classification/idmr_{criterion}.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser(description='Input Distribution-Matching Regularization')
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
    print(args)
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
