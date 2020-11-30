
import os, sys, pickle
from tabulate import tabulate

result_folder = sys.argv[1]
header = ['filename', 'seed', 'type', 'bs', 'me', 'pa', '#lines', 'best']
all_runs = set([fn[:-4] for fn in os.listdir(result_folder) if fn[-4:] in ['.arg', '.log']])
infos = []
for run in all_runs:
    args = pickle.load(open(f'{result_folder}/{run}.arg', 'rb'))
    lines = open(f'{result_folder}/{run}.log').readlines()
    aucs = [float(l.strip().split('\t')[3]) for l in lines]
    info = [run, args.model_seed, args.evaluation_type, args.batchsize, args.max_epoch, args.patience,
            len(lines), f'{max(aucs):0.3f}']
    infos.append(info)
infos = sorted(infos, key=lambda s:s[1:])
print(tabulate(infos, header, tablefmt='grid'))
