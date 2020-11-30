
import argparse, random, time, string, pickle, os
from argparse import Namespace
from copy import deepcopy as copy
from tqdm import trange
import numpy as np
from train_scheduler import TrainScheduler
import help_text as ht

def swap_kernel(order, args, internal_swp_prob=0.5):
    order = copy(order)
    N_pool = len(order)
    if random.random() < internal_swp_prob:
        i, j = random.sample(range(args.tot_acq), 2)
        while int(i / args.batchsize) == int(j / args.batchsize):
            i, j = random.sample(range(args.tot_acq), 2)
        order[i], order[j] = order[j], order[i]
    else:
        i = random.randint(0, args.tot_acq - 1)
        j = random.randint(args.tot_acq, N_pool - 1)
        order[i], order[j] = order[j], order[i]
    return order

def new_args_result_file(folder):
    unique_code = ''.join(random.choices(string.ascii_lowercase, k=8))
    args_fn = f'{folder}/{unique_code}.arg'
    log_fn = f'{folder}/{unique_code}.log'
    while os.path.isfile(args_fn):
        unique_code = ''.join(random.choices(string.ascii_lowercase, k=8))
        args_fn = f'{folder}/{unique_code}.arg'
        log_fn = f'{folder}/{unique_code}.log'
    return args_fn, log_fn

class Runner():
    def __init__(self, args):
        self.args = args
        self.args.evaluation_set = 'valid'
        self.N_pool = 1000
        args_fn, log_fn = new_args_result_file(args.log_dir)
        with open(args_fn, 'wb') as args_f:
            pickle.dump(args, args_f, protocol=0)  # protocol 0 for human readable format
        self.log_file = open(log_fn, 'w')
        self.runned = False

    def evaluate_order(self, order):
        curve = self.train_scheduler.evaluate_order(order)
        quality = np.mean(curve)
        return curve, quality

    def log(self, order, curve, quality, time, T):
        string = f'{order[:self.args.tot_acq]} \t {curve} \t time:{time:0.3f}|T:{T} \t {quality}'
        self.log_file.write(string + '\n')
        self.log_file.flush()

    def run(self):
        assert not self.runned, 'The runner has been runned before'
        self.runned = True
        self.train_scheduler = TrainScheduler(self.args)
        print('train scheduler initialized')
        start_time = time.time()
        order = random.sample(range(self.N_pool), self.N_pool)
        curve, quality = self.evaluate_order(order)
        best_quality = quality
        best_order = order
        self.log(order, curve, quality, time.time() - start_time, 'N/A')
        print('Simulated annealing search')
        for i in trange(self.args.num_sa_samples):
            T = (i + 1) * self.args.anneal_factor
            start_time = time.time()
            new_order = swap_kernel(order, self.args)
            new_curve, new_quality = self.evaluate_order(new_order)
            ratio = np.exp((new_quality - quality) * T)
            if random.random() < ratio:
                order = new_order
                curve = new_curve
                quality = new_quality
                if quality > best_quality:
                    best_quality = quality
                    best_order = order
            tot_time = time.time() - start_time
            self.log(order, curve, quality, tot_time, T)
        order = best_order
        quality = best_quality
        print('Greedy search')
        for i in trange(self.args.num_greedy_samples):
            start_time = time.time()
            new_order = swap_kernel(order, self.args)
            new_curve, new_quality = self.evaluate_order(new_order)
            if new_quality > quality:
                order = new_order
                curve = new_curve
                quality = new_quality
            tot_time = time.time() - start_time
            self.log(order, curve, quality, tot_time, 'greedy')

    def __del__(self):
        self.log_file.close()

def main(model_seed=0, data_seed=0, batchsize=25, max_epoch=20, patience=20, tot_acq=250, log_dir='logs',
         anneal_factor=0.1, num_sa_samples=40000, num_greedy_samples=5000, use_gpus='all', workers_per_gpu=1):
    args = Namespace(model_seed=model_seed, data_seed=data_seed, batchsize=batchsize, max_epoch=max_epoch, patience=patience,
                     tot_acq=tot_acq, log_dir=log_dir, anneal_factor=anneal_factor, num_sa_samples=num_sa_samples,
                     num_greedy_samples=num_greedy_samples, use_gpus=use_gpus, workers_per_gpu=workers_per_gpu)
    runner = Runner(args)
    runner.run()

def main_cli():
    parser = argparse.ArgumentParser(description='Optimal order search')
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=250, help=ht.tot_acq)
    parser.add_argument('--log-dir', type=str, default='logs', help=ht.log_dir)
    parser.add_argument('--anneal-factor', type=float, default=0.1, help=ht.anneal_factor)
    parser.add_argument('--num-sa-samples', type=int, default=40000, help=ht.num_sa_samples)
    parser.add_argument('--num-greedy-samples', type=int, default=5000, help=ht.num_greedy_samples)
    parser.add_argument('--use-gpus', type=str, default='all', help=ht.use_gpus)
    parser.add_argument('--workers-per-gpu', type=int, default=1, help=ht.workers_per_gpu)
    args = parser.parse_args()
    print(args)
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
