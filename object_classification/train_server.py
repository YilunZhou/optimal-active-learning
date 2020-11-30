
import sys, socket, random, argparse, time

import numpy as np
import torch

from trainer import get_trainer
from utils import load_data

parser = argparse.ArgumentParser(description='Training server')
parser.add_argument('--port', type=int)
parser.add_argument('--model-seed', type=int)
parser.add_argument('--data-seed', type=int)
parser.add_argument('--batchsize', type=int)
parser.add_argument('--max-epoch', type=int)
parser.add_argument('--patience', type=int)
parser.add_argument('--gpu-idx', type=int)
parser.add_argument('--evaluation-set', type=str)
args = parser.parse_args()

'''
client --> server communication is a list of integers, with all but the last one representing the order
of the pool, and the last one representing the number of data points to use from the pool
server --> client communication is a string for the floating point validation metric
both communications end with <EOS>
'''

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((socket.gethostname(), args.port))
sock.listen(1)
print(f'server listening on {socket.gethostname()} {args.port}')
dataset = load_data(args.data_seed)

model_sel_set = dataset['model_sel']
model_sel_X, model_sel_y = map(np.array, zip(*model_sel_set))
model_sel_X = torch.tensor(model_sel_X).float().to(f'cuda:{args.gpu_idx}')
model_sel_y = torch.tensor(model_sel_y).long().to(f'cuda:{args.gpu_idx}')

eval_set = dataset[args.evaluation_set]
eval_X, eval_y = map(np.array, zip(*eval_set))
eval_X = torch.tensor(eval_X).float().to(f'cuda:{args.gpu_idx}')
eval_y = torch.tensor(eval_y).long().to(f'cuda:{args.gpu_idx}')

def server_train(order, num_use_pool):
    effective_pool = list([dataset['pool'][o] for o in order[:num_use_pool]])
    train_set = dataset['warmstart'] + effective_pool
    train_X, train_y = map(np.array, zip(*train_set))
    train_X = torch.tensor(train_X).float().to(f'cuda:{args.gpu_idx}')
    train_y = torch.tensor(train_y).long().to(f'cuda:{args.gpu_idx}')
    trainer = get_trainer(args.model_seed, f'cuda:{args.gpu_idx}', mcdropout=False)
    trainer.train((train_X, train_y), (model_sel_X, model_sel_y), args.batchsize, args.max_epoch, args.patience,
                  test_steps=None, verbose=False)
    metric = trainer.evaluate_acc(trainer.best_model, eval_X, eval_y)
    return metric

while True:
    connection, client_address = sock.accept()
    spec = ''
    while True:
        data = connection.recv(16)
        if data:
            spec = spec + data.decode('utf-8')
        if spec.endswith('<EOS>'):
            spec = list(map(int, spec.replace('<EOS>', '').strip().split()))
            order = spec[:-1]
            num_use_pool = spec[-1]
            assert num_use_pool % args.batchsize == 0

            metric = server_train(order, num_use_pool)
            ret_msg = bytearray(f'{metric} <EOS>', 'utf-8')
            spec = ''
            connection.sendall(ret_msg)
    raise Exception('Another connection? ')
