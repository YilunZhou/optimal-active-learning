
import random, pickle, math, time, argparse, shelve
from copy import deepcopy as copy
from tqdm import trange
import numpy as np
import torch

from trainer import get_trainer
from utils import store_baseline
import help_text as ht

def acquire(model, pool_dict, criterion, batchsize):
    assert criterion in ['min-confidence', 'normalized-min-confidence', 'longest'], \
                        f'Unknown criterion {criterion}'
    assert isinstance(pool_dict, dict), 'pool_dict needs to be a dictionary'
    idxs = list(pool_dict.keys())
    data = list(pool_dict.values())
    sents, _ = zip(*data)
    lens = [len(s) for s in sents]
    if criterion == 'longest':
        return [idxs[i] for i in np.argsort(lens)[::-1][:batchsize]]
    else:
        with torch.no_grad():
            logits, _ = model.tag(sents)
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        max_log_probs = log_probs.max(axis=-1)
        max_log_probs = [list(p[:l].flat) for l, p in zip(lens, max_log_probs)]
        tot_log_probs = [sum(p) for p in max_log_probs]
        if criterion == 'normalized-min-confidence':
            tot_log_probs = [t / l for t, l in zip(tot_log_probs, lens)]
        combo = sorted(list(zip(tot_log_probs, idxs)))
        return [c[1] for c in combo[:batchsize]]

def main(criterion, evaluation_set, model_seed=0, data_seed=0, batchsize=25, max_epoch=100, patience=20,
         tot_acq=250, gpu_idx=0):
    data = pickle.load(open('data/restaurant.pkl', 'rb'))['seeds'][data_seed]
    train_set = copy(data['warmstart'])
    model_sel_set = copy(data['train_valid'])
    pool_dict = {i: p for i, p in enumerate(data['pool'])}
    eval_set = data[evaluation_set]
    eval_sents, eval_tags = zip(*eval_set)
    curve = []
    data_order = []
    trainer = get_trainer(model_seed, device=f'cuda:{gpu_idx}')
    trainer.train(train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False)
    f1 = trainer.evaluate_f1(trainer.best_model, eval_sents, eval_tags)
    curve.append(f1)
    for _ in trange(int(tot_acq / batchsize)):
        acquire_idxs = acquire(trainer.best_model, pool_dict, criterion, batchsize)
        data_order.extend(acquire_idxs)
        for idx in acquire_idxs:
            train_set.append(pool_dict[idx])
            del pool_dict[idx]
        trainer = get_trainer(model_seed, device=f'cuda:{gpu_idx}')
        trainer.train(train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False)
        f1 = trainer.evaluate_f1(trainer.best_model, eval_sents, eval_tags)
        curve.append(f1)
    print(curve)
    print(np.mean(curve))
    store_baseline(curve, data_order, criterion, evaluation_set, model_seed, data_seed,
                   batchsize, max_epoch, patience, tot_acq)

def main_cli():
    parser = argparse.ArgumentParser('Heuristic baselines')
    ########################### required args ############################
    parser.add_argument('--criterion', type=str, help=ht.criterion)
    parser.add_argument('--evaluation-set', type=str, help=ht.evaluation_set)
    ######################################################################
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=250, help=ht.tot_acq)
    parser.add_argument('-gpu-idx', type=int, default=0, help=ht.gpu_idx)
    args = parser.parse_args()
    print(args)
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
