
import shelve, argparse, os
import numpy as np
import matplotlib.pyplot as plt
import help_text as ht

def main(model='lstm', model_seeds=[0, 1, 2, 3, 4], domain='alarm', data_seed=0,
         batchsize=20, max_epoch=100, patience=20, tot_acq=160):
    qualities = np.zeros((len(model_seeds), len(model_seeds)))
    quality_gaps = np.zeros((len(model_seeds), len(model_seeds)))
    with shelve.open('statistics/perf_curves.shv') as curves:
        for i, m_to in enumerate(model_seeds):
            for j, m_from in enumerate(model_seeds):
                spec_trans = f'{model} {m_from} {m_to} {domain} {data_seed} {batchsize} {max_epoch} { patience} {tot_acq}'
                spec_native = f'{model} {m_to} {m_to} {domain} {data_seed} {batchsize} {max_epoch} { patience} {tot_acq}'
                qualities[i, j] = np.mean(curves[spec_trans]['test'])
                quality_gaps[i, j] = np.mean(curves[spec_trans]['test']) - np.mean(curves[spec_native]['test'])
    plt.imshow(quality_gaps, cmap='coolwarm', vmin=-abs(quality_gaps).max(), vmax=abs(quality_gaps).max())
    for i in range(len(model_seeds)):
        for j in range(len(model_seeds)):
            plt.annotate(f'{qualities[i, j]:0.3f}', xy=(j, i), ha='center', va='center', fontsize=13)
            rect = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, linewidth=1)
            plt.gca().add_patch(rect)
    # plt.colorbar()
    plt.xticks(range(len(model_seeds)), [str(s) if s != -1 else 'rand' for s in model_seeds], fontsize=13)
    plt.yticks(range(len(model_seeds)), [str(s) if s != -1 else 'rand' for s in model_seeds], fontsize=13)
    plt.xlabel('Source Seed')
    plt.ylabel('Target Seed')
    plt.title('Intent Classification', fontsize=13)
    plt.savefig('../figures/intent_classification/seed_transfer.pdf', bbox_inches='tight')

def main_cli():
    parser = argparse.ArgumentParser(description='Plot seed transfer quality matrix. ')
    parser.add_argument('--model', type=str, default='lstm', help=ht.model)
    parser.add_argument('--model-seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help=ht.model_seeds)
    parser.add_argument('--domain', type=str, default='alarm', help=ht.domain)
    parser.add_argument('--data-seed', type=int, default=0, help=ht.data_seed)
    parser.add_argument('--tot-acq', type=int, default=160, help=ht.tot_acq)
    parser.add_argument('--batchsize', type=int, default=20, help=ht.batchsize)
    parser.add_argument('--max-epoch', type=int, default=100, help=ht.max_epoch)
    parser.add_argument('--patience', type=int, default=20, help=ht.patience)
    args = parser.parse_args()
    print(args)
    main(**vars(args))


if __name__ == '__main__':
    main_cli()
