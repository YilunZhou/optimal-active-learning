
import os, pickle, time, argparse, shelve
from copy import deepcopy as copy
from collections import Counter
from tqdm import trange

import numpy as np
from scipy.stats import entropy

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from trainer import get_trainer, roberta_expand_data, roberta_get_tokenizer
from roberta_model_store import roberta_get_model
from utils import store_baseline

import help_text as ht

def score_bald(model, pool_sents):
    torch.manual_seed(0)
    assert model.mcdropout is True, 'MC dropout not enabled on BALD?'
    with torch.no_grad():
        preds = []
        for _ in range(100):
            pool_logits = model(pool_sents, evall=False).cpu().numpy()
            pred = pool_logits.argmax(axis=1)
            preds.append(pred)
    preds = np.array(preds)
    cts = [Counter(list(preds[:, i].flat)) for i in range(len(pool_sents))]
    majorities = [ct.most_common(1)[0][1] for ct in cts]
    scores = majorities  # the smaller the majority number is, the more disagreeing it is
    return scores

def score_maxent(model, pool_sents):
    assert model.mcdropout is False, 'MC dropout enabled on Max-Entropy?'
    with torch.no_grad():
        pool_probs = F.softmax(model(pool_sents, evall=True), dim=1).cpu().numpy()
    scores = - entropy(pool_probs, axis=1)
    return scores

def active_learn(criterion, evaluation_set, model, model_seed, domain, data_seed,
                 batchsize, max_epoch, patience, tot_acq, gpu_idx):
    data = pickle.load(open('data/TOP.pkl', 'rb'))[domain]['seeds'][data_seed]
    train_set = copy(data['warmstart'])
    pool_dict = {i: p for i, p in enumerate(data['pool'])}
    model_sel_set = data['train_valid']
    eval_set = data[evaluation_set]
    curve = []
    mcdropout = (criterion == 'bald')

    trainer = get_trainer(model, model_seed, domain, f'cuda:{gpu_idx}', mcdropout=mcdropout)
    trainer.train(train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False)
    f1 = trainer.evaluate_f1(trainer.best_model, eval_set)
    curve.append(f1)
    data_order = []
    for _ in trange(int(tot_acq / batchsize)):
        pool_idxs = list(pool_dict.keys())
        pool_sents, _ = zip(*pool_dict.values())
        if criterion == 'bald':
            pool_scores = score_bald(trainer.best_model, pool_sents)
        elif criterion == 'max-entropy':
            pool_scores = score_maxent(trainer.best_model, pool_sents)
        scores_idxs = sorted(zip(pool_scores, pool_idxs))
        for _, idx in scores_idxs[:batchsize]:
            train_set.append(pool_dict[idx])
            data_order.append(idx)
            del pool_dict[idx]
        trainer = get_trainer(model, model_seed, domain, f'cuda:{gpu_idx}', mcdropout=mcdropout)
        trainer.train(train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False)
        f1 = trainer.evaluate_f1(trainer.best_model, eval_set)
        curve.append(f1)
    return curve, data_order

def score_bald_roberta(model, pool_t, pool_m, gpu_idx):
    assert model.mcdropout is True, 'MC dropout not enabled on BALD?'
    with torch.no_grad():
        model.eval()
        preds = []
        for _ in range(100):
            logits, = model(input_ids=torch.tensor(pool_t).long().to(gpu_idx),
                            attention_mask=torch.tensor(pool_m).float().to(gpu_idx), evall=False)
            pred = logits.cpu().numpy().argmax(axis=1)
            preds.append(pred)
        model.train()
    preds = np.array(preds)
    cts = [Counter(list(preds[:, i].flat)) for i in range(len(pool_t))]
    majorities = [ct.most_common(1)[0][1] for ct in cts]
    scores = majorities  # the smaller the majority number is, the more disagreeing it
    return scores

def score_maxent_roberta(model, pool_t, pool_m, gpu_idx):
    assert model.mcdropout is False, 'MC dropout enabled on Max-Entropy?'
    with torch.no_grad():
        model.eval()
        logits, = model(input_ids=torch.tensor(pool_t).long().to(gpu_idx),
                        attention_mask=torch.tensor(pool_m).float().to(gpu_idx), evall=True)
        model.train()
        pool_probs = F.softmax(logits, dim=1).cpu().numpy()
        scores = - entropy(pool_probs, axis=1)
    return scores

def active_learn_roberta(criterion, evaluation_set, model, model_seed, domain, data_seed,
                 batchsize, max_epoch, patience, tot_acq, gpu_idx):
    tokenizer = roberta_get_tokenizer()
    data = pickle.load(open('data/TOP.pkl', 'rb'))[domain]['seeds'][data_seed]
    wp_t, wp_m, wp_l = roberta_expand_data(data['warmstart'] + data['pool'], tokenizer)
    warmstart_t = wp_t[:len(data['warmstart'])]
    pool_t = wp_t[len(data['warmstart']):]
    warmstart_m = wp_m[:len(data['warmstart'])]
    pool_m = wp_m[len(data['warmstart']):]
    warmstart_l = wp_l[:len(data['warmstart'])]
    pool_l = wp_l[len(data['warmstart']):]
    warmstart = (warmstart_t, warmstart_m,warmstart_l)
    model_sel_set = roberta_expand_data(data['train_valid'], tokenizer)
    eval_set = roberta_expand_data(data[evaluation_set], tokenizer)
    train_t, train_m, train_l = map(list, copy(warmstart))
    curve = []
    mcdropout = (criterion == 'bald')
    trainer = get_trainer(model, model_seed, domain, f'cuda:{gpu_idx}', mcdropout=mcdropout)
    trainer.train((train_t, train_m, train_l), model_sel_set, batchsize, max_epoch, patience, verbose=False)
    f1 = trainer.evaluate_f1(trainer.best_model, eval_set, batchsize)
    curve.append(f1)
    pool_dict = {i: c for i, c in enumerate(zip(pool_t, pool_m, pool_l))}
    data_order = []
    for _ in trange(int(tot_acq / batchsize)):
        pool_idxs = list(pool_dict.keys())
        pool_t, pool_m, _ = zip(*pool_dict.values())
        assert len(pool_t) == len(pool_m) == len(pool_idxs)
        if criterion == 'bald':
            pool_scores = score_bald_roberta(trainer.best_model, pool_t, pool_m, gpu_idx)
        elif criterion == 'max-entropy':
            pool_scores = score_maxent_roberta(trainer.best_model, pool_t, pool_m, gpu_idx)
        scores_idxs = sorted(zip(pool_scores, pool_idxs))
        for s, i in scores_idxs[:batchsize]:
            assert len(pool_dict[i]) == 3
            train_t.append(pool_dict[i][0])
            train_m.append(pool_dict[i][1])
            train_l.append(pool_dict[i][2])
            data_order.append(i)
            del pool_dict[i]
        trainer = get_trainer(model, model_seed, domain, f'cuda:{gpu_idx}', mcdropout=mcdropout)
        trainer.train((train_t, train_m, train_l), model_sel_set, batchsize, max_epoch, patience, verbose=False)
        f1 = trainer.evaluate_f1(trainer.best_model, eval_set, batchsize)
        curve.append(f1)
    return curve, data_order

def main(criterion, evaluation_set, model, model_seed, domain='alarm', data_seed=0, batchsize=20, max_epoch=100, patience=20,
         tot_acq=160, gpu_idx=0):
    if model != 'roberta':
        curve, order = active_learn(criterion, evaluation_set, model, model_seed, domain, data_seed,
                                    batchsize, max_epoch, patience, tot_acq, gpu_idx)
    else:
        curve, order = active_learn_roberta(criterion, evaluation_set, model, model_seed, domain, data_seed,
                                    batchsize, max_epoch, patience, tot_acq, gpu_idx)
    print(f'Peformance curve {curve}, quality {np.mean(curve):0.3f}')
    store_baseline(curve, order, criterion, evaluation_set, model, model_seed, domain, data_seed,
                   batchsize, max_epoch, patience, tot_acq)

def main_cli():
    parser = argparse.ArgumentParser(description='Construct heuristic baselines. ')
    ################## required args ##################
    parser.add_argument('--criterion', type=str, choices=['max-entropy', 'bald'], help=ht.criterion)
    parser.add_argument('--evaluation-set', type=str, help=ht.evaluation_set)
    parser.add_argument('--model', type=str, choices=['lstm', 'cnn', 'aoe', 'roberta'], help=ht.model)
    parser.add_argument('--model-seed', type=int, help=ht.model_seed)
    ###################################################
    parser.add_argument('--domain', type=str, default='alarm', help=ht.domain)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=20, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=160, help=ht.tot_acq)
    parser.add_argument('--gpu-idx', type=int, default=0, help=ht.gpu_idx)
    args = parser.parse_args()
    main(**vars(args))


if __name__ == '__main__':
    main_cli()
