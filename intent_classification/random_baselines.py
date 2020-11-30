
import argparse, random, time, shelve
from argparse import Namespace
from tqdm import trange
import numpy as np
from train_scheduler import TrainScheduler
import help_text as ht
from utils import store_baseline

def main(evaluation_set, model, model_seed, domain='alarm', data_seed=0,
         batchsize=20, max_epoch=100, patience=20, tot_acq=160,
         use_gpus='all', workers_per_gpu=1, num_random_samples=100):
        N_pool = 800
        train_args = Namespace(evaluation_set=evaluation_set, model=model, model_seed=model_seed,
                               domain=domain, data_seed=data_seed, batchsize=batchsize, max_epoch=max_epoch, patience=patience,
                               tot_acq=tot_acq, use_gpus=use_gpus, workers_per_gpu=workers_per_gpu)
        scheduler = TrainScheduler(train_args)
        order = list(range(N_pool))
        curves = []
        for _ in trange(num_random_samples):
            random.shuffle(order)
            curve = scheduler.evaluate_order(order)
            curves.append(curve)
        avg_curve = np.mean(curves, axis=0)
        store_baseline(avg_curve, None, 'random', evaluation_set, model, model_seed, domain, data_seed,
                       batchsize, max_epoch, patience, tot_acq)

def main_cli():
    parser = argparse.ArgumentParser(description='Generate random baselines')
    #################### required args ####################
    parser.add_argument('--evaluation-set', type=str, help=ht.evaluation_set)
    parser.add_argument('--model', type=str, help=ht.model)
    parser.add_argument('--model-seed', type=int, help=ht.model_seed)
    ########################################################

    parser.add_argument('--domain', type=str, default='alarm', help=ht.domain)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=20, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=160, help=ht.tot_acq)

    parser.add_argument('--use-gpus', type=str, default='all', help=ht.use_gpus)
    parser.add_argument('--workers-per-gpu', type=int, default=1, help=ht.workers_per_gpu)
    parser.add_argument('--num-random-samples', type=int, default=100, help=ht.num_random_samples)

    args = parser.parse_args()
    print(args)
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
