
import socket, sys, argparse, subprocess, time, random
from multiprocessing.pool import ThreadPool
import torch

def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def create_server(args, gpu_idx):
    '''start a train_server process on a free port, and return the port number'''
    port = get_open_port()
    python_args = ['python', 'train_server.py',
                   '--port', str(port),
                   '--model-seed', str(args.model_seed),
                   '--data-seed', str(args.data_seed),
                   '--evaluation-set', str(args.evaluation_set),
                   '--batchsize', str(args.batchsize),
                   '--patience', str(args.patience),
                   '--max-epoch', str(args.max_epoch),
                   '--gpu-idx', str(gpu_idx),
                  ]
    process = subprocess.Popen(python_args)
    return process, port

class Client:
    def __init__(self, remote_port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((socket.gethostname(), remote_port))
        self.in_use = False

    def query(self, order, num_used):
        assert not self.in_use, 'client currently in use?'
        self.in_use = True
        spec = ' '.join(map(str, order + [num_used])) + '<EOS>'
        self.sock.sendall(bytearray(spec, encoding='utf-8'))
        entire_data = ''
        while True:
            data = self.sock.recv(16)
            entire_data = entire_data + data.decode('utf-8')
            if entire_data.endswith('<EOS>'):
                break
        metric = float(entire_data.replace('<EOS>', ''))
        self.in_use = False
        return metric

class TrainScheduler:
    def __init__(self, args):
        '''
        args required fields:
        workers_per_gpu, domain, data_seed, model_seed, evaluation_set
        bootstrap, val_metric, patience, max_epoch, batchsize
        '''
        self.args = args
        self.processes = []
        self.ports = []
        self.workers_per_gpu = args.workers_per_gpu
        if args.use_gpus == 'all':
            self.use_gpus = list(range(torch.cuda.device_count()))
        else:
            self.use_gpus = list(map(int, args.use_gpus.split(',')))
        for gpu_idx in self.use_gpus * args.workers_per_gpu:
            process, port = create_server(args, gpu_idx)
            self.processes.append(process)
            self.ports.append(port)
            time.sleep(0.1)
        time.sleep(10)
        self.clients = []
        for port in self.ports:
            self.clients.append(Client(port))
        self.cache = dict()

    def get_free_client(self):
        time.sleep(random.random() * 0.05)
        in_uses = [c.in_use for c in self.clients]
        while sum(in_uses) == len(in_uses):
            print('waiting in use')
            time.sleep(random.random() * 0.05 + 0.02)
            in_uses = [c.in_use for c in self.clients]
        idx = in_uses.index(False)
        return idx

    def query(self, order, num_used):
        idx = self.get_free_client()
        return self.clients[idx].query(order, num_used)

    def evaluate_order(self, order):
        num_used = list(range(0, self.args.tot_acq + 1, self.args.batchsize))
        results = [None] * len(num_used)
        query_args = []
        miss_idxs = []
        for i, idx in enumerate(num_used):
            cur_order = tuple(order[:idx])
            if cur_order in self.cache:
                results[i] = self.cache[cur_order]
            else:
                miss_idxs.append(i)
                query_args.append((order, idx))
        pool = ThreadPool(len(self.use_gpus) * self.workers_per_gpu)
        metrics = pool.starmap(self.query, query_args)
        for mi, m, q in zip(miss_idxs, metrics, query_args):
            assert results[mi] is None
            results[mi] = m
            cur_order = tuple(order[:q[1]])
            assert cur_order not in self.cache
            self.cache[cur_order] = m
        assert None not in results
        pool.close()
        pool.join()
        return results

    def __del__(self):
        for p in self.processes:
            p.kill()
