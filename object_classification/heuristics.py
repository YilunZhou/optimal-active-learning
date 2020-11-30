
import argparse, pickle, random, shelve, math
from tqdm import trange

import numpy as np
from scipy.stats import entropy
from batchbald_redux import batchbald
import torch

from trainer import get_trainer
from utils import load_data, store_baseline
import help_text as ht

torch.backends.cudnn.deterministic = True

def acquire(model, pool_dict, criterion, batchsize, gpu_idx, num_inference_steps, num_joint_entropy_samples):
    idxs, imgs_ys = list(pool_dict.keys()), list(pool_dict.values())
    imgs, ys = zip(*imgs_ys)
    imgs = torch.tensor(np.stack(imgs, axis=0)).float().to(gpu_idx)
    if criterion == 'max-entropy':
        with torch.no_grad():
            probs = np.exp(model(imgs).cpu().numpy())
        scores = - entropy(probs, axis=1)
        score_idxs = np.argsort(scores)[:batchsize]
        return [idxs[i] for i in score_idxs]
    elif criterion in ['bald', 'batchbald']:
        with torch.no_grad():
            model.eval()
            logprobs = []
            bs = 100
            num_batches = math.ceil(imgs.shape[0] / bs)
            for b_idx in range(num_batches):
                cur_imgs = imgs[b_idx * bs : (b_idx + 1) * bs]
                logprobs.append(model(cur_imgs, num_inference_steps))
            logprobs = torch.cat(logprobs, dim=0)
            logprobs = logprobs.double()
            if criterion == 'batchbald':
                candidate_batch = batchbald.get_batchbald_batch(logprobs.exp(), batchsize, num_joint_entropy_samples,
                                                                dtype=torch.double, device=gpu_idx)
            else:
                candidate_batch = batchbald.get_bald_batch(logprobs.exp(), batchsize,
                                                           dtype=torch.double, device=gpu_idx)
        return [idxs[i] for i in candidate_batch.indices]

def main(criterion, evaluation_set='test', model_seed=0, data_seed=0, batchsize=25, max_epoch=100, patience=20, tot_acq=300,
         gpu_idx=0, num_inference_steps=100, num_joint_entropy_samples=100000, num_eval_steps=5):
    data = load_data(data_seed)
    warmstart = data['warmstart']
    warmstart_X, warmstart_y = map(np.array, zip(*warmstart))
    pool = data['pool']
    model_sel = data['model_sel']
    model_sel_X, model_sel_y = map(np.array, zip(*model_sel))
    model_sel_X = torch.tensor(model_sel_X).float().to(gpu_idx)
    model_sel_y = torch.tensor(model_sel_y).long().to(gpu_idx)
    eval_set = data[evaluation_set]
    eval_X, eval_y = map(np.array, zip(*eval_set))
    eval_X = torch.tensor(eval_X).float().to(gpu_idx)
    eval_y = torch.tensor(eval_y).long().to(gpu_idx)

    pool_dict = {i: (img, y) for i, (img, y) in enumerate(pool)}
    train_X = warmstart_X.copy()
    train_y = warmstart_y.copy()

    if criterion in ['bald', 'batchbald']:
        mcdropout = True
    else:
        mcdropout = False
        num_eval_steps = None
    mcdropout = ('bald' in criterion)  # True when criterion is 'bald' or 'batchbald', False otherwise
    curve = []
    trainer = get_trainer(model_seed, gpu_idx, mcdropout)
    trainer.train((torch.tensor(train_X).float().to(gpu_idx), torch.tensor(train_y).long().to(gpu_idx)),
                  (model_sel_X, model_sel_y), batchsize, max_epoch, patience, num_eval_steps, verbose=False)
    acc = trainer.evaluate_acc(trainer.best_model, eval_X, eval_y, num_eval_steps)
    curve.append(acc)
    data_order = []
    for _ in trange(int(tot_acq / batchsize)):
        idxs = acquire(trainer.best_model, pool_dict, criterion, batchsize,
                       gpu_idx, num_inference_steps, num_joint_entropy_samples)
        data_order = data_order + idxs
        new_X = np.stack([pool_dict[i][0] for i in idxs], axis=0)
        new_y = np.array([pool_dict[i][1] for i in idxs])
        train_X = np.concatenate((train_X, new_X), axis=0)
        train_y = np.concatenate((train_y, new_y), axis=0)
        for idx in idxs:
            del pool_dict[idx]
        trainer = get_trainer(model_seed, gpu_idx, mcdropout)
        trainer.train((torch.tensor(train_X).float().to(gpu_idx), torch.tensor(train_y).long().to(gpu_idx)),
                    (model_sel_X, model_sel_y), batchsize, max_epoch, patience, num_eval_steps, verbose=False)
        acc = trainer.evaluate_acc(trainer.best_model, eval_X, eval_y, num_eval_steps)
        curve.append(acc)
    print(curve)
    print(np.mean(curve))
    store_baseline(curve, data_order, criterion, evaluation_set,
                   model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)

def main_cli():
    parser = argparse.ArgumentParser()
    ############################# required args #############################
    parser.add_argument('--criterion', type=str, help=ht.criterion)
    #########################################################################
    parser.add_argument('--evaluation-set', type=str, default='test', help=ht.evaluation_set)
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=300, help=ht.tot_acq)
    parser.add_argument('--gpu-idx', type=int, default=0, help=ht.gpu_idx)
    parser.add_argument('--num-inference-steps', type=int, default=100, help=ht.num_inference_steps)
    parser.add_argument('--num-joint-entropy-samples', type=int, default=100000, help=ht.num_joint_entropy_samples)
    parser.add_argument('--num-eval-steps', type=int, default=5, help=ht.num_eval_steps)
    args = parser.parse_args()
    print(args)
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
