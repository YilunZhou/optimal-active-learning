
import argparse
from collections import Counter
from tqdm import trange

import numpy as np
from copy import deepcopy as copy
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

from trainer import get_trainer
from utils import load_data, load_optimal, load_baseline, store_baseline
from performance_curve import plot_curves
from distribution_vis import plot_label_proportion, plot_ref_meter
import help_text as ht

def prioritize_labels(cur_labels, label_ref):
    num_labels = 10
    N_pool = len(cur_labels)
    expected = [p * N_pool for p in label_ref]
    actual_ct = Counter(cur_labels)
    actual = [actual_ct[i] for i in range(num_labels)]
    deficits = np.array(actual) - expected
    assert min(deficits) <= 0
    labels = np.argsort(deficits)
    return labels

def select_batch(scores, idxs, use_labels, true_labels, train_labels, label_ref, batchsize):
    true_labels = {i: tl for i, tl in zip(idxs, true_labels)}
    scores_idxs_labels = sorted(zip(scores, idxs, use_labels))
    use_idxs = []
    cur_labels = copy(list(train_labels))
    for _ in range(batchsize):
        target_labels = prioritize_labels(cur_labels, label_ref)
        for tgt_lbl in target_labels:
            found = False
            for (s, i, lbl) in scores_idxs_labels:
                if lbl == tgt_lbl and i not in use_idxs:
                    use_idxs.append(i)
                    cur_labels.append(true_labels[i])
                    found = True
                    break
            if found:
                break
    assert len(set(use_idxs)) == len(use_idxs) == batchsize
    return use_idxs

def odmr(data, evaluation_set, label_ref, imputed, model_seed,
         batchsize, max_epoch, patience, tot_acq, gpu_idx, smoothing=1):
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
    num_labels = 10
    assert label_ref in ['accessible', 'test']
    assert imputed in [True, False]
    if label_ref == 'accessible':
        _, warmstart_labels = zip(*data['warmstart'])
        _, model_sel_labels = zip(*data['model_sel'])
        labels = list(warmstart_labels) + list(model_sel_labels)
    else:
        _, labels = zip(*data[evaluation_set])
    ct = Counter(labels)
    label_ref = np.array([ct[i] + smoothing for i in range(num_labels)]) / (len(labels) + smoothing * num_labels)

    train_set = copy(data['warmstart'])
    pool_set = copy(data['pool'])
    pool_dict = {i: p for i, p in enumerate(pool_set)}
    model_sel_set = data['model_sel']
    model_sel_X, model_sel_y = zip(*model_sel_set)
    model_sel_X = torch.tensor(model_sel_X).float().to(gpu_idx)
    model_sel_y = torch.tensor(model_sel_y).long().to(gpu_idx)
    eval_X, eval_y = zip(*data[evaluation_set])
    eval_X = torch.tensor(eval_X).float().to(gpu_idx)
    eval_y = torch.tensor(eval_y).long().to(gpu_idx)

    curves = []
    train_X, train_y = zip(*train_set)
    train_X = torch.tensor(train_X).float().to(gpu_idx)
    train_y = torch.tensor(train_y).long().to(gpu_idx)
    trainer = get_trainer(model_seed, f'cuda:{gpu_idx}', False)
    trainer.train((train_X, train_y), (model_sel_X, model_sel_y), batchsize, max_epoch,
                  patience, None, verbose=False)
    acc = trainer.evaluate_acc(trainer.best_model, eval_X, eval_y, None)
    curves.append(acc)
    data_order = []
    for _ in trange(int(tot_acq / batchsize)):
        idxs = list(pool_dict.keys())
        pool_X, pool_y = zip(*pool_dict.values())
        with torch.no_grad():
            pool_X = torch.tensor(pool_X).float().to(gpu_idx)
            pool_probs = np.exp(trainer.best_model(pool_X).cpu().numpy())
        scores = - entropy(pool_probs, axis=1)
        pool_pred_labels = pool_probs.argmax(axis=1)
        pool_true_labels = pool_y
        _, train_labels = zip(*train_set)
        if imputed:
            use_labels = pool_true_labels
        else:
            use_labels = pool_pred_labels
        use_idxs = select_batch(scores, idxs, use_labels, pool_true_labels, train_labels, label_ref, batchsize)
        train_set = train_set + [pool_dict[i] for i in use_idxs]
        data_order = data_order + use_idxs
        for idx in use_idxs:
            del pool_dict[idx]

        train_X, train_y = zip(*train_set)
        train_X = torch.tensor(train_X).float().to(gpu_idx)
        train_y = torch.tensor(train_y).long().to(gpu_idx)
        trainer = get_trainer(model_seed, f'cuda:{gpu_idx}', False)
        trainer.train((train_X, train_y), (model_sel_X, model_sel_y), batchsize,
                      max_epoch, patience, None, verbose=False)
        acc = trainer.evaluate_acc(trainer.best_model, eval_X, eval_y, None)
        curves.append(acc)
    return curves, data_order


def main(model_seed=0, data_seed=0, batchsize=25, max_epoch=100, patience=20, tot_acq=300,
         evaluation_set='test', gpu_idx=0, log_dir='logs'):
    data = load_data(data_seed)
    num_labels = 10

    N_warmstart = len(data['warmstart'])
    N_pool = len(data['pool'])

    _, warmstart_y = zip(*data['warmstart'])
    _, pool_y = zip(*data['pool'])
    _, eval_y = zip(*data[evaluation_set])

    try:
        l1 = load_baseline(f'odmr-l1-max-entropy', evaluation_set, model_seed, data_seed,
                           batchsize, max_epoch, patience, tot_acq)
        l1_curve, l1_order = l1['curve'], l1['order']
        print(l1_curve, np.mean(l1_curve))
        l2 = load_baseline(f'odmr-l2-max-entropy', evaluation_set, model_seed, data_seed,
                           batchsize, max_epoch, patience, tot_acq)
        l2_curve, l2_order = l2['curve'], l2['order']
        print(l2_curve, np.mean(l2_curve))
        l3 = load_baseline(f'odmr-l3-max-entropy', evaluation_set, model_seed, data_seed,
                           batchsize, max_epoch, patience, tot_acq)
        l3_curve, l3_order = l3['curve'], l3['order']
        print(l3_curve, np.mean(l3_curve))
        l4 = load_baseline(f'odmr-l4-max-entropy', evaluation_set, model_seed, data_seed,
                           batchsize, max_epoch, patience, tot_acq)
        l4_curve, l4_order = l4['curve'], l4['order']
        print(l4_curve, np.mean(l4_curve))
    except KeyError:
        l1_curve, l1_order = odmr(data, evaluation_set, 'test', True, model_seed, batchsize, max_epoch, patience,
                                tot_acq, gpu_idx, smoothing=1)
        print(l1_curve, np.mean(l1_curve))
        store_baseline(l1_curve, l1_order, f'odmr-l1-max-entropy', evaluation_set, model_seed,
                       data_seed, batchsize, max_epoch, patience, tot_acq)
        l2_curve, l2_order = odmr(data, evaluation_set, 'accessible', True, model_seed, batchsize, max_epoch, patience,
                                tot_acq, gpu_idx, smoothing=1)
        print(l2_curve, np.mean(l2_curve))
        store_baseline(l2_curve, l2_order, f'odmr-l2-max-entropy', evaluation_set, model_seed,
                       data_seed, batchsize, max_epoch, patience, tot_acq)
        l3_curve, l3_order = odmr(data, evaluation_set, 'test', False, model_seed, batchsize, max_epoch, patience,
                                tot_acq, gpu_idx, smoothing=1)
        print(l3_curve, np.mean(l3_curve))
        store_baseline(l3_curve, l3_order, f'odmr-l3-max-entropy', evaluation_set, model_seed,
                       data_seed, batchsize, max_epoch, patience, tot_acq)
        l4_curve, l4_order = odmr(data, evaluation_set, 'accessible', False, model_seed, batchsize, max_epoch, patience,
                                tot_acq, gpu_idx, smoothing=1)
        print(l4_curve, np.mean(l4_curve))
        store_baseline(l4_curve, l4_order, f'odmr-l4-max-entropy', evaluation_set, model_seed,
                       data_seed, batchsize, max_epoch, patience, tot_acq)

    plt.figure(figsize=[20, 4])
    gs = GridSpec(ncols=9, nrows=1, width_ratios=[10, 0.3, 10, 0.7, 10, 0.7, 10, 0.7, 10], wspace=0.05)
    plt.subplot(gs[0, 0])
    baselines = [('max-entropy', 'Max-Entropy', 0),
                 ('odmr-l1-max-entropy', 'Test + True', 1),
                 ('odmr-l2-max-entropy', 'Acce + True', 2),
                 ('odmr-l3-max-entropy', 'Test + Pred', 6),
                 ('odmr-l4-max-entropy', 'Acce + Pred', 8),
                 ('random', 'Random', 4)]
    xs = list(range(N_warmstart, N_warmstart + tot_acq + 1, batchsize))
    optimal_order, optimal_quality, _ = load_optimal(log_dir, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)
    plot_curves(optimal_order, xs, evaluation_set, model_seed, model_seed, data_seed,
                batchsize, max_epoch, patience, tot_acq, None, None, baselines)
    plt.xlabel('# Data Points')
    plt.ylabel('Accuracy')
    plt.title('Object Classification')

    plt.subplot(gs[0, 2])
    plot_label_proportion(l1_order, warmstart_y, pool_y, eval_y)
    plt.title('Test + Groundtruth')
    plt.xlabel('# Data Points')
    plt.yticks([])

    plt.subplot(gs[0, 3])
    plot_ref_meter(eval_y)

    plt.subplot(gs[0, 4])
    plot_label_proportion(l2_order, warmstart_y, pool_y, eval_y)
    plt.title('Accessible + Groundtruth')
    plt.xlabel('# Data Points')
    plt.yticks([])

    plt.subplot(gs[0, 5])
    plot_ref_meter(eval_y)

    plt.subplot(gs[0, 6])
    plot_label_proportion(l3_order, warmstart_y, pool_y, eval_y)
    plt.title('Test + Predicted')
    plt.xlabel('# Data Points')
    plt.yticks([])

    plt.subplot(gs[0, 7])
    plot_ref_meter(eval_y)

    plt.subplot(gs[0, 8])
    plot_label_proportion(l4_order, warmstart_y, pool_y, eval_y)
    plt.title('Accessible + Predicted')
    plt.xlabel('# Data Points')
    plt.yticks([])

    plt.savefig('../figures/object_classification/odmr.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser(description='ODMR with Max-Entropy Heuristic')
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--tot-acq', type=int, default=300, help=ht.tot_acq)
    parser.add_argument('--evaluation-set', type=str, default='test', help=ht.evaluation_set)
    parser.add_argument('--gpu-idx', type=int, default=0, help=ht.gpu_idx)
    parser.add_argument('--log-dir', type=str, default='logs', help=ht.log_dir)
    args = parser.parse_args()
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
