
import sys, socket, random, argparse, pickle, time
from sklearn.metrics import f1_score

import numpy as np
import torch

from trainer import get_trainer

parser = argparse.ArgumentParser(description='Training server')
parser.add_argument('--port', type=int)
parser.add_argument('--model-seed', type=int)
parser.add_argument('--data-seed', type=int)
parser.add_argument('--batchsize', type=int)
parser.add_argument('--max-epoch', type=int)
parser.add_argument('--patience', type=int)
parser.add_argument('--evaluation-set', type=str)  # valid or test
parser.add_argument('--gpu-idx', type=int)
args = parser.parse_args()

'''
client --> server communication is a list of integers, with all but the last one representing the order
of the pool, and the last one representing the number of data points to use from the pool
server --> client communication is a string for the floating point validation metric
both communications end with <EOS>
'''

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((socket.gethostname(), args.port))
sock.listen(1)
print(f'server listening on {socket.gethostname()} {args.port}')

dataset = pickle.load(open('data/restaurant.pkl', 'rb'))['seeds'][args.data_seed]
eval_set = dataset[args.evaluation_set]

def server_train(ordering, num_use_pool):
    effective_pool = list([dataset['pool'][o] for o in ordering[:num_use_pool]])
    train_set = dataset['warmstart'] + effective_pool
    model_sel_set = dataset['train_valid']
    trainer = get_trainer(model_seed=args.model_seed, device=f'cuda:{args.gpu_idx}')
    trainer.train(train_set, model_sel_set, args.batchsize, args.max_epoch, args.patience, verbose=False)
    eval_sents, eval_labels = zip(*eval_set)
    metric = trainer.evaluate_f1(trainer.best_model, eval_sents, eval_labels)
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
            ordering = spec[:-1]
            num_use_pool = spec[-1]
            metric = server_train(ordering, num_use_pool)
            ret_msg = bytearray(f'{metric} <EOS>', 'utf-8')
            spec = ''
            connection.sendall(ret_msg)
    raise Exception('Another connection? ')
