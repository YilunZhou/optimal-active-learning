
import random, pickle, time, os, shelve, shutil
from copy import deepcopy as copy
import numpy as np


def abs_path(p):
    return os.path.join(os.path.dirname(__file__), p)

def load_data(data_seed):
    if not os.path.isfile(abs_path('data/fashion_mnist.pkl')):
        concat_data_shards()
    dataset = pickle.load(open(abs_path('data/fashion_mnist.pkl'), 'rb'))['seeds'][data_seed]
    return dataset

def concat_data_shards():
    if os.path.isfile(abs_path('data/fashion_mnist.pkl')):
        return
    with open(abs_path('data/fashion_mnist.pkl'), 'wb') as wfd:
        for i in range(10):
            with open(abs_path(f'data/fashion_mnist.part.{i}'), 'rb') as fd:
                shutil.copyfileobj(fd, wfd)

def load_optimal(log_dir, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq):
    all_runs = set([fn[:-4] for fn in os.listdir(log_dir) if fn[-4:] in ['.arg', '.log']])
    best_quality = -1
    for run in all_runs:
        args = pickle.load(open(f'{log_dir}/{run}.arg', 'rb'))
        if not [model_seed, data_seed, batchsize, max_epoch, patience, tot_acq] == \
                    [args.model_seed, args.data_seed, args.batchsize, args.max_epoch, args.patience, args.tot_acq]:
            continue
        for l in open(f'{log_dir}/{run}.log'):
            order, curve, _, quality = l.strip().split('\t')
            quality = float(quality)
            if best_quality < quality:
                best_quality = quality
                best_order = list(map(int, order.strip()[1:-1].split(',')))
                best_curve = curve
    if best_quality == -1:
        raise Exception('No valid runs have been found. ')
    best_curve = list(map(float, best_curve.strip()[1:-1].split(',')))
    return best_order, best_quality, best_curve

def load_baseline(name, eval_set, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq):
    spec = ' '.join(map(str, [name, eval_set, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq]))
    with shelve.open(abs_path('statistics/baselines.shv')) as baselines:
        return baselines[spec]

def store_baseline(curve, order, name, eval_set, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq):
    if name == 'random':
        assert order is None, 'order needs to be None for random baseline'
    spec = f'{name} {eval_set} {model_seed} {data_seed} {batchsize} {max_epoch} {patience} {tot_acq}'
    with shelve.open(abs_path('statistics/baselines.shv')) as baselines:
        baselines[spec] = {'curve': curve, 'order': order}
