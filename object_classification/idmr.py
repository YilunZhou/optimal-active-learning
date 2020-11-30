
import argparse, pickle, shelve, math
from collections import Counter
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch

from trainer import get_trainer
from performance_curve import plot_curves
from utils import load_data, load_optimal, load_baseline, store_baseline
import help_text as ht

def sort(model, pool_dict, gpu_idx):
    idxs, imgs_ys_clsts = list(pool_dict.keys()), list(pool_dict.values())
    imgs, ys, _ = zip(*imgs_ys_clsts)
    imgs = torch.tensor(np.stack(imgs, axis=0)).float().to(gpu_idx)
    with torch.no_grad():
        probs = np.exp(model(imgs).cpu().numpy())
    scores = - entropy(probs, axis=1)
    score_idxs = np.argsort(scores)
    return [idxs[i] for i in score_idxs]

def largest_deficit(train_clst, ref, num_clusters):
    N = len(train_clst)
    expected = [ref[c] * N for c in range(num_clusters)]
    actual = Counter(train_clst)
    deficit = [actual[c] - e for c, e in enumerate(expected)]
    return np.argsort(deficit)[0]

def idmr(data, evaluation_set, model_seed, batchsize, max_epoch, patience, tot_acq, tsne_dim, num_clusters, gpu_idx):
    warmstart = data['warmstart']
    warmstart_X, warmstart_y = map(np.array, zip(*warmstart))
    pool = data['pool']
    pool_X, pool_y = map(np.array, zip(*pool))
    model_sel = data['model_sel']
    model_sel_X, model_sel_y = map(np.array, zip(*model_sel))

    accessible_X = np.vstack((np.array(warmstart_X).reshape(len(warmstart_X), -1),
                              np.array(model_sel_X).reshape(len(model_sel_X), -1),
                              np.array(pool_X).reshape(len(pool_X), -1)))
    accessible_y = np.concatenate([warmstart_y, model_sel_y, pool_y])
    pca = PCA(n_components=100, random_state=np.random.RandomState(0))
    tsne = TSNE(n_components=tsne_dim, n_jobs=-1, random_state=np.random.RandomState(0))
    accessible_X_red = tsne.fit_transform(pca.fit_transform(accessible_X))
    warmstart_X_red, model_sel_X_red, pool_X_red = np.array_split(accessible_X_red,
                                                     [len(warmstart_X), len(warmstart_X) + len(model_sel_X)])
    kmeans = KMeans(n_clusters=num_clusters, random_state=np.random.RandomState(0))
    accessible_clst = kmeans.fit_predict(accessible_X_red)
    warmstart_clst, model_sel_clst, pool_clst = np.array_split(accessible_clst,
                                                      [len(warmstart_X), len(warmstart_X) + len(model_sel_X)])
    ref_ct = Counter(accessible_clst)
    ref_dist = [ref_ct[cl] / len(accessible_clst) for cl in range(num_clusters)]
    eval_set = data[evaluation_set]
    eval_X, eval_y = map(np.array, zip(*eval_set))

    model_sel_X = torch.tensor(model_sel_X).float().to(gpu_idx)
    model_sel_y = torch.tensor(model_sel_y).long().to(gpu_idx)
    eval_X = torch.tensor(eval_X).float().to(gpu_idx)
    eval_y = torch.tensor(eval_y).long().to(gpu_idx)

    pool_dict = {i: (img, y, cl) for i, ((img, y), cl) in enumerate(zip(pool, pool_clst))}
    train_X = warmstart_X.copy()
    train_y = warmstart_y.copy()
    train_clst = list(warmstart_clst.copy().flat)
    data_order = []
    curve = []
    trainer = get_trainer(model_seed, gpu_idx, False)
    trainer.train((torch.tensor(train_X).float().to(gpu_idx), torch.tensor(train_y).long().to(gpu_idx)),
                  (model_sel_X, model_sel_y), batchsize, max_epoch, patience, None, verbose=False)
    acc = trainer.evaluate_acc(trainer.best_model, eval_X, eval_y, None)
    curve.append(acc)
    for _ in trange(int(tot_acq / batchsize)):
        sorted_idxs = sort(trainer.best_model, pool_dict, gpu_idx)
        use_idxs = []
        for _ in range(batchsize):
            target = largest_deficit(train_clst, ref_dist, num_clusters)
            for idx in sorted_idxs:
                if idx not in use_idxs and pool_dict[idx][2] == target:
                    use_idxs.append(idx)
                    train_clst.append(target)
                    break
        data_order = data_order + use_idxs
        new_X = np.stack([pool_dict[i][0] for i in use_idxs], axis=0)
        new_y = np.array([pool_dict[i][1] for i in use_idxs])
        train_X = np.concatenate((train_X, new_X), axis=0)
        train_y = np.concatenate((train_y, new_y), axis=0)
        for idx in use_idxs:
            del pool_dict[idx]
        trainer = get_trainer(model_seed, gpu_idx, False)
        trainer.train((torch.tensor(train_X).float().to(gpu_idx), torch.tensor(train_y).long().to(gpu_idx)),
                      (model_sel_X, model_sel_y), batchsize, max_epoch, patience, None, verbose=False)
        acc = trainer.evaluate_acc(trainer.best_model, eval_X, eval_y, None)
        curve.append(acc)
    return curve, data_order

def main(model_seed=0, data_seed=0, batchsize=25, max_epoch=100, patience=20, tot_acq=300, evaluation_set='test',
         log_dir='logs', tsne_dim=3, num_clusters=5, gpu_idx=0):
    data = load_data(data_seed)
    N_warmstart = len(data['warmstart'])
    try:
        idmr_curve = load_baseline('idmr-max-entropy', evaluation_set, model_seed, data_seed,
                                   batchsize, max_epoch, patience, tot_acq)['curve']
    except KeyError:
        data = load_data(data_seed)
        idmr_curve, idmr_order = idmr(data, evaluation_set, model_seed, batchsize, max_epoch, patience, tot_acq,
                                      tsne_dim, num_clusters, gpu_idx)
        store_baseline(idmr_curve, idmr_order, 'idmr-max-entropy', evaluation_set, model_seed, data_seed,
                    batchsize, max_epoch, patience, tot_acq)
    print(idmr_curve)
    print(np.mean(idmr_curve))

    plt.figure()
    xs = list(range(N_warmstart, N_warmstart + tot_acq + 1, batchsize))
    baselines = [('max-entropy', 'Max-Entropy', 0), ('bald', 'BALD', 1), ('random', 'Random', 4),
                 ('idmr-max-entropy', 'IDMR Max-Ent.', 6)]
    optimal_order, _, _ = load_optimal(log_dir, model_seed, data_seed, batchsize, max_epoch, patience, tot_acq)
    plot_curves(optimal_order, xs, evaluation_set, model_seed, model_seed, data_seed,
                batchsize, max_epoch, patience, tot_acq, None, None, baselines)
    plt.title('IDMR Performance Curve')
    plt.savefig('../figures/object_classification/idmr.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser('IDMR with Max-Entropy Heuristic')
    parser.add_argument('--model-seed', type=int, default=0, help=ht.model_seed)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--batchsize', type=int, default=25, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    parser.add_argument('--tot-acq', type=int, default=300, help=ht.tot_acq)
    parser.add_argument('--evaluation-set', type=str, default='test', help=ht.evaluation_set)
    parser.add_argument('--log-dir', type=str, default='logs', help=ht.log_dir)
    parser.add_argument('--tsne-dim', type=int, default=3, help=ht.tsne_dim)
    parser.add_argument('--num-clusters', type=int, default=5, help=ht.num_clusters)
    parser.add_argument('--gpu-idx', type=int, default=0, help=ht.gpu_idx)
    args = parser.parse_args()
    print(args)
    main(**vars(args))

if __name__ == '__main__':
    main_cli()
