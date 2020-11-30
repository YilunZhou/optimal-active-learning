
import sys, socket, random, argparse, pickle, time

import numpy as np
import torch

from trainer import get_trainer, roberta_expand_data, roberta_get_tokenizer
from roberta_model_store import roberta_get_model

parser = argparse.ArgumentParser(description='Training server')
parser.add_argument('--port', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--model-seed', type=int)
parser.add_argument('--domain', type=str)
parser.add_argument('--data-seed', type=int)
parser.add_argument('--batchsize', type=int)
parser.add_argument('--max-epoch', type=int)
parser.add_argument('--patience', type=int)
parser.add_argument('--evaluation-set', type=str)
parser.add_argument('--gpu-idx', type=int)

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
data = pickle.load(open('data/TOP.pkl', 'rb'))[args.domain]['seeds'][args.data_seed]

if args.model != 'roberta':
    warmstart_set = data['warmstart']
    pool_set = data['pool']
    model_sel_set = data['train_valid']
    eval_set = data[args.evaluation_set]
    def server_train(order, num_use_pool):
        train_set = warmstart_set + [pool_set[o] for o in order[:num_use_pool]]
        trainer = get_trainer(args.model, args.model_seed, args.domain, f'cuda:{args.gpu_idx}')
        model, _ = trainer.train(train_set, model_sel_set, args.batchsize, args.max_epoch, args.patience, False)
        metric = trainer.evaluate_f1(model, eval_set)
        return metric
else:
    import logging
    logging.basicConfig(level=logging.ERROR)
    tokenizer = roberta_get_tokenizer()
    wp_t, wp_m, wp_l = roberta_expand_data(data['warmstart'] + data['pool'], tokenizer)
    warmstart_t = wp_t[:len(data['warmstart'])]
    pool_t = wp_t[len(data['warmstart']):]
    warmstart_m = wp_m[:len(data['warmstart'])]
    pool_m = wp_m[len(data['warmstart']):]
    warmstart_l = wp_l[:len(data['warmstart'])]
    pool_l = wp_l[len(data['warmstart']):]
    model_sel_set = roberta_expand_data(data['train_valid'], tokenizer)
    eval_set = roberta_expand_data(data[args.evaluation_set], tokenizer)
    def server_train(order, num_use):
        use_t = list(warmstart_t) + [pool_t[o] for o in order[:num_use]]
        use_m = list(warmstart_m) + [pool_m[o] for o in order[:num_use]]
        use_l = list(warmstart_l) + [pool_l[o] for o in order[:num_use]]
        train_set = (use_t, use_m, use_l)
        trainer = get_trainer(args.model, args.model_seed, args.domain, f'cuda:{args.gpu_idx}')
        model, _ = trainer.train(train_set, model_sel_set, args.batchsize, args.max_epoch, args.patience, False)
        f1 = trainer.evaluate_f1(model, eval_set, args.batchsize)
        return f1

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
