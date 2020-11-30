
import os, pickle, time, argparse, shelve
from copy import deepcopy as copy
from argparse import Namespace
from collections import Counter
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
import torch

from trainer import get_trainer
from utils import load_optimal, load_baseline, store_baseline
from performance_curve import plot_curves
import help_text as ht

def score_model(model, pool_dict, criterion, batchsize):
    assert criterion in ['min-confidence', 'normalized-min-confidence', 'longest'], \
                        f'Unknown criterion {criterion}'
    assert isinstance(pool_dict, dict), 'pool_dict needs to be a dictionary'
    idxs, data = list(pool_dict.keys()), list(pool_dict.values())
    sents, _ = zip(*data)
    lens = [len(s) for s in sents]
    if criterion == 'longest':
        scores = [-l for l in lens]
        return {i: s for i, s in zip(idxs, scores)}
    else:
        with torch.no_grad():
            logits, _ = model.tag(sents)
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        max_log_probs = log_probs.max(axis=-1)
        max_log_probs = [list(p[:l].flat) for l, p in zip(lens, max_log_probs)]
        tot_log_probs = [sum(p) for p in max_log_probs]
        if criterion == 'normalized-min-confidence':
            tot_log_probs = [t / l for t, l in zip(tot_log_probs, lens)]
        return {i: s for i, s in zip(idxs, tot_log_probs)}

def prioritize_len_groups(cur_lens, lens_proportions):
    len_groups, ref_props = list(lens_proportions.keys()), list(lens_proportions.values())
    len_groups_dict = {v: i for i, vs in enumerate(len_groups) for v in range(vs[0], vs[1]+1)}
    N = len(cur_lens)
    expected = [p * N for p in ref_props]
    cur_len_groups = [len_groups_dict[l] for l in cur_lens]
    cur_len_groups_ct = Counter(cur_len_groups)
    deficits = [cur_len_groups_ct[lg] - e for lg, e in zip(range(len(len_groups)), expected)]
    assert min(deficits) <= 0
    gs = np.argsort(deficits)
    return [len_groups[g] for g in gs]

def select_batch(scores, idxs, lens, train_lens, lens_proportions, batchsize):
    scores_idxs_lens = sorted(zip(scores, idxs, lens))
    use_idxs = []
    cur_lens = copy(train_lens)
    for _ in range(batchsize):
        lgs = prioritize_len_groups(cur_lens, lens_proportions)
        for lo, hi in lgs:
            found = False
            for (s, i, l) in scores_idxs_lens:
                if lo <= l <= hi and i not in use_idxs:
                    use_idxs.append(i)
                    cur_lens.append(l)
                    found = True
                    break
            if found:
                break
    assert len(set(use_idxs)) == len(use_idxs) == batchsize
    return use_idxs

def idmr(data, evaluation_set, criterion, lens_proportions, model_seed,
         batchsize, max_epoch, patience, tot_acq, gpu_idx):
    '''
    lens_proportions is a dictionary from (len_low, len_high), both inclusive, to reference proportions.
    the acquisition order will try to preserve this proportion in the warmstart + acquired set
    '''
    train_set = copy(data['warmstart'])
    pool_set = copy(data['pool'])
    pool_dict = {i: p for i, p in enumerate(pool_set)}
    model_sel_set = data['train_valid']
    eval_sents, eval_labels = zip(*data[evaluation_set])
    curve = []

    trainer = get_trainer(model_seed, device=f'cuda:{gpu_idx}')
    trainer.train(train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False)
    f1 = trainer.evaluate_f1(trainer.best_model, eval_sents, eval_labels)
    curve.append(f1)
    data_order = []
    for _ in trange(int(tot_acq / batchsize)):
        idxs = list(pool_dict.keys())
        pool_sents, _ = zip(*pool_dict.values())
        scores = score_model(trainer.best_model, pool_dict, criterion, batchsize)
        scores = [scores[i] for i in idxs]
        pool_lens = [len(p) for p in pool_sents]
        train_lens = [len(ts[0]) for ts in train_set]
        use_idxs = select_batch(scores, idxs, pool_lens, train_lens, lens_proportions, batchsize)
        train_set = train_set + [pool_dict[i] for i in use_idxs]
        data_order = data_order + use_idxs
        for idx in use_idxs:
            del pool_dict[idx]
        trainer = get_trainer(model_seed, device=f'cuda:{gpu_idx}')
        trainer.train(train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False)
        f1 = trainer.evaluate_f1(trainer.best_model, eval_sents, eval_labels)
        curve.append(f1)
    return curve, data_order

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

def main(criterion, evaluation_set, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq, log_dir, gpu_idx):
    data = pickle.load(open('data/restaurant.pkl', 'rb'))['seeds'][data_seed]
    N_warmstart = len(data['warmstart'])
    try:
        idmr_curve = load_baseline(f'idmr-{criterion}', evaluation_set, model_seed, data_seed,
                                   batchsize, max_epoch, patience, tot_acq)['curve']
    except KeyError:
        N_pool = len(data['pool'])
        accessible_set = data['warmstart'] + data['train_valid'] + data['pool']
        accessible_sents, _ = zip(*accessible_set)
        lens = [len(s) for s in accessible_sents]
        lens_ct = Counter(lens)
        max_len = max(lens_ct.keys())
        cts = np.array([lens_ct[l] for l in range(max_len + 1)])
        props = cts / sum(cts)
        groups = group_proportions(props, 0.13)
        lens_proportions = dict()
        for i, g in enumerate(groups):
            lo, hi = min(g), max(g)
            sum_props = sum(props[p] for p in range(lo, hi + 1))
            if i == len(groups) - 1:
                hi = 100
            lens_proportions[(lo, hi)] = sum_props
        idmr_curve, idmr_order = idmr(data, evaluation_set, criterion, lens_proportions, model_seed,
                                    batchsize, max_epoch, patience, tot_acq, gpu_idx)
        store_baseline(idmr_curve, idmr_order, f'idmr-{criterion}', evaluation_set, model_seed, data_seed,
                    batchsize, max_epoch, patience, tot_acq)
    print(idmr_curve)
    print(np.mean(idmr_curve))

    plt.figure()
    optimal_order, _, _ = load_optimal(log_dir, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)
    xs = list(range(N_warmstart, N_warmstart + tot_acq + 1, batchsize))
    display_name = {'min-confidence': 'Min-Confidence', 'normalized-min-confidence': 'Norm.-Min-Conf.',
                    'longest': 'Longest'}[criterion]
    baselines = [(criterion, display_name, 0), (f'idmr-{criterion}', f'IDMR-{display_name.replace("idence", ".")}', 6),
                 ('random', 'Random', 4)]
    plot_curves(optimal_order, xs, evaluation_set, model_seed, model_seed, data_seed, batchsize, max_epoch, patience,
                tot_acq, None, None, baselines)
    plt.tight_layout()
    plt.savefig(f'../figures/named_entity_recognition/idmr_{criterion}.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser(description='Input Distribution-Matching Regularization')
    ########################### required args ############################
    parser.add_argument('--criterion', type=str, help=ht.criterion)
    ######################################################################
    parser.add_argument('--evaluation-set', type=str, default='test', help=ht.evaluation_set)
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=250, help=ht.tot_acq)
    parser.add_argument('--log-dir', type=str, default='logs', help=ht.log_dir)
    parser.add_argument('--gpu-idx', type=int, default=0, help=ht.gpu_idx)
    args = parser.parse_args()
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
