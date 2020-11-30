
import sys, random
from copy import deepcopy as copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from batchbald_redux import consistent_mc_dropout

class BayesianCNN(consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)
        return input

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1(input), 2))
        input = F.relu(F.max_pool2d(self.conv2(input), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1(input))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)
        return input

class Trainer():
    def __init__(self, model, device):
        self.model = model
        self.model.to(device)
        self.loss = nn.NLLLoss()
        self.optimizer = Adam(self.model.parameters())
        self.device = device
        self.trained = False

    def train_batch(self, images, labels):
        self.model.train()
        self.model.zero_grad()
        if isinstance(self.model, CNN):
            logprobs = self.model(images)
        elif isinstance(self.model, BayesianCNN):
            logprobs = self.model(images, 1).squeeze(1)
        loss_val = self.loss(logprobs, labels)
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()

    def train(self, train_set, valid_set, batchsize, max_epoch, patience, test_steps=None, verbose=False):
        '''
        train the model for max_epoch, and validate after every epoch.
        if the validation metric is not improving for patience epochs, early-terminate the training
        and return the best model
        patience == 1 means terminating the training as soon as the validation metric stops improving
        '''
        assert self.trained is False, 'This trainer has been trained before'
        self.trained = True
        device = self.device
        train_X, train_y = train_set
        assert isinstance(train_X, torch.Tensor) and isinstance(train_y, torch.Tensor)
        valid_X, valid_y = valid_set
        assert isinstance(valid_X, torch.Tensor) and isinstance(valid_y, torch.Tensor)
        assert len(train_y) % batchsize == 0, 'train_set size must be a multiple of batch size'
        n_batches = int(len(train_X) / batchsize)
        best_valid_metric = -1
        cur_patience = 0
        for epoch in range(max_epoch):
            losses = []
            for i in range(n_batches):
                cur_X = train_X[i * batchsize : (i + 1) * batchsize]
                cur_y = train_y[i * batchsize : (i + 1) * batchsize]
                loss = self.train_batch(cur_X, cur_y)
                losses.append(loss)
            cur_patience += 1
            valid_metric = self.evaluate_acc(self.model, valid_X, valid_y, test_steps)
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                best_model = copy(self.model)
                best_epoch = epoch
                cur_patience = 0
            elif cur_patience == patience:
                    break
            if verbose:
                print(f'epoch {epoch}: loss {np.mean(losses):0.3f}, val metric {valid_metric:0.3f}')
        self.best_model = best_model
        return best_model, best_valid_metric

    @staticmethod
    def evaluate_acc(model, images, labels, steps=None):
        model.eval()
        if isinstance(model, BayesianCNN):
            with torch.no_grad():
                predictions = model(images, steps)
                predictions = torch.logsumexp(predictions, dim=1)
                preds = predictions.max(1)[1].cpu().numpy()
        elif isinstance(model, CNN):
            with torch.no_grad():
                predictions = model(images).cpu().numpy()
                preds = predictions.argmax(axis=1)
        model.train()
        labels = labels.cpu().numpy()
        return (preds == labels).mean()

def get_trainer(model_seed, device, mcdropout):
    torch.backends.cudnn.deterministic = True
    if model_seed >= 0:
        torch.manual_seed(model_seed)
    else:
        assert model_seed == -1, 'The only alllowed negative model seed is -1'
    if mcdropout:
        model = BayesianCNN(num_classes=10)
    else:
        model = CNN(num_classes=10)
    torch.manual_seed(random.randint(0, sys.maxsize))
    trainer = Trainer(model, device)
    return trainer
