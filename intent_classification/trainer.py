
import pickle, sys, random, os, math
from copy import deepcopy as copy

import numpy as np
from sklearn.metrics import f1_score
import nltk

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from transformers import RobertaTokenizer
from roberta_model_store import roberta_get_model

class BiLSTMModel(nn.Module):
    def __init__(self, num_classes, word_embedding_data, tokenize_func, mcdropout=False):
        '''
        BiLSTM model for top-level intent classification
        tokenize_func takes a sentence string and returns a list of tokens for each word in the sentence,
        as well as the list of tokenized words. unknown word is given the token value == num-tokens
        '''
        super(BiLSTMModel, self).__init__()
        num_embedding, emb_dim = word_embedding_data.shape
        self.unk_idx = num_embedding
        self.pad_idx = num_embedding + 1
        word_embedding_data = torch.cat((torch.tensor(word_embedding_data).float(),
                                         torch.randn(1, emb_dim) * 0.01,             # <unk>
                                         torch.zeros(1, emb_dim)                     # <pad>
                                        ), dim=0)
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding_data, freeze=False,
                                                      padding_idx=self.pad_idx)
        self.tokenize_func = tokenize_func
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim,
                            batch_first=True, bidirectional=True)
        self.output = nn.Linear(emb_dim * 2, num_classes)
        self.mcdropout = mcdropout

    def forward(self, sentences, evall):
        '''sentences is a list of untokenized strings'''
        device = next(self.parameters()).device
        list_of_tokens, list_of_words = zip(*map(self.tokenize_func, sentences))
        lens = list(map(len, list_of_tokens))
        max_len = max(lens)
        list_of_embeddings = []
        for tks, words in zip(list_of_tokens, list_of_words):
            pad_len = max_len - len(tks)
            if pad_len == 0:
                word_embs = self.word_embedding(torch.tensor(tks).to(device))
            else:
                pad_tks = (torch.ones(pad_len) * self.pad_idx).long()
                cat_tks = torch.cat((torch.tensor(tks), pad_tks)).to(device)
                word_embs = self.word_embedding(cat_tks)
            embs = word_embs
            list_of_embeddings.append(embs)
        embeddings_batch = torch.stack(list_of_embeddings, dim=0)
        embeddings_pack = pack_padded_sequence(embeddings_batch, torch.tensor(lens),
                                                batch_first=True, enforce_sorted=False)
        lstm_hidden_pack, (_, _) = self.lstm(embeddings_pack)
        lstm_hidden, unpacked_lens = pad_packed_sequence(lstm_hidden_pack, batch_first=True)
        avg_hiddens = []
        for hidden, l in zip(lstm_hidden, lens):
            avg_hidden = hidden[:l].mean(dim=0)
            avg_hiddens.append(avg_hidden)
        avg_hiddens = torch.stack(avg_hiddens, dim=0)
        if self.mcdropout and not evall:
            avg_hiddens = F.dropout(avg_hiddens, p=0.5, training=True)
        logits = self.output(avg_hiddens)
        return logits


class CNNModel(nn.Module):
    def __init__(self, num_classes, word_embedding_data, tokenize_func, mcdropout=False):
        super(CNNModel, self).__init__()
        torch.backends.cudnn.enabled = False
        num_embedding, emb_dim = word_embedding_data.shape
        self.unk_idx = num_embedding
        self.pad_idx = num_embedding + 1
        word_embedding_data = torch.cat((torch.tensor(word_embedding_data).float(),
                                         torch.randn(1, emb_dim) * 0.01,             # <unk>
                                         torch.zeros(1, emb_dim)                     # <pad>
                                        ), dim=0)
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding_data, freeze=False,
                                                      padding_idx=self.pad_idx)
        self.tokenize_func = tokenize_func
        self.convs = nn.Sequential(nn.Conv1d(emb_dim, 50, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv1d(50, 50, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv1d(50, 50, kernel_size=3, padding=1),
                                   nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(50, 50),
                                 nn.ReLU(),
                                )
        self.output = nn.Linear(50, num_classes)
        self.mcdropout = mcdropout

    def forward(self, sentences, evall):
        '''sentences is a list of untokenized strings'''
        device = next(self.parameters()).device
        list_of_tokens, _ = zip(*map(self.tokenize_func, sentences))
        lens = list(map(len, list_of_tokens))
        max_len = max(lens)
        embeddings = []
        for tks in list_of_tokens:
            padded_tks = tks + [self.pad_idx] * (max_len - len(tks))
            word_embs = self.word_embedding(torch.tensor(padded_tks).to(device))
            embeddings.append(word_embs)
        embeddings_batch = torch.stack(embeddings, dim=0).permute(0, 2, 1)
        conv_tensor = self.convs(embeddings_batch)
        avg_tensors = torch.stack([conv_tensor[i, :, :l].mean(dim=1) for i, l in enumerate(lens)], dim=0)
        hidden = self.mlp(avg_tensors)
        if self.mcdropout and not evall:
            hidden = F.dropout(hidden, p=0.5, training=True)
        logits = self.output(hidden)
        return logits


class AOEModel(nn.Module):
    def __init__(self, num_classes, word_embedding_data, tokenize_func, mcdropout=False):
        '''bag (average) of embedding followed by a couple fully connected layers'''
        super(AOEModel, self).__init__()
        num_embedding, emb_dim = word_embedding_data.shape
        self.unk_idx = num_embedding
        self.pad_idx = num_embedding + 1
        word_embedding_data = torch.cat((torch.tensor(word_embedding_data).float(),
                                         torch.randn(1, emb_dim) * 0.01,             # <unk>
                                         torch.zeros(1, emb_dim)                     # <pad>
                                        ), dim=0)
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding_data, freeze=False,
                                                      padding_idx=self.pad_idx)
        self.tokenize_func = tokenize_func
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 100),
                                 nn.ReLU(),
                                )
        self.output = nn.Linear(100, num_classes)
        self.mcdropout = mcdropout

    def forward(self, sentences, evall):
        '''sentences is a list of untokenized strings'''
        device = next(self.parameters()).device
        list_of_tokens, _ = zip(*map(self.tokenize_func, sentences))
        lens = list(map(len, list_of_tokens))
        max_len = max(lens)
        list_of_aoes = []
        for tks in list_of_tokens:
            word_embs = self.word_embedding(torch.tensor(tks).to(device))
            list_of_aoes.append(word_embs.mean(dim=0))
        embeddings_batch = torch.stack(list_of_aoes, dim=0)
        hidden = self.mlp(embeddings_batch)
        if self.mcdropout and not evall:
            hidden = F.dropout(hidden, p=0.5, training=True)
        logits = self.output(hidden)
        return logits


class Trainer():
    def __init__(self, model, device):
        self.model = model
        self.model.to(device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters())
        self.device = device
        self.trained = False

    def train_batch(self, batch):
        self.model.train()
        self.model.zero_grad()
        sents, labels = zip(*batch)
        labels = torch.tensor(labels).to(self.device)
        logits = self.model(sents, evall=False)
        loss_val = self.loss(logits, labels)
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()

    def train(self, train_set, model_sel_set, batchsize, max_epoch, patience, verbose=False):
        '''
        train the model for max_epoch, and validate after every epoch.
        if the validation metric is not improving for patience epochs, early-terminate the training
        and return the best model
        patience == 1 means terminating the training as soon as the validation metric stops improving
        train_func(train_batch) performs a single batch update on the model
        valid_func(model, model_sel_set) returns the validation metric (e.g. accuracy or loss) of the model
        a higher validation metric value is better
        '''
        assert not self.trained, 'This trainer has been trained'
        self.trained = True
        assert len(train_set) % batchsize == 0, 'train set size must be a multiple of batch size'
        n_batches = int(len(train_set) / batchsize)
        best_metric = float('-inf')
        cur_patience = 0
        for epoch in range(max_epoch):
            losses = []
            for i in range(n_batches):
                cur_batch = train_set[i * batchsize : (i + 1) * batchsize]
                loss = self.train_batch(cur_batch)
                losses.append(loss)
            metric = self.evaluate_f1(self.model, model_sel_set)
            if verbose:
                print(f'epoch {epoch}: loss {np.mean(losses):0.3f}, val metric {metric:0.3f}')
            if metric > best_metric:
                best_metric = metric
                best_model = copy(self.model)
                cur_patience = 0
            else:
                cur_patience += 1
                if cur_patience == patience:
                    break
        try:
            best_model.lstm.flatten_parameters()
        except:
            pass
        self.best_model = best_model
        return best_model, best_metric

    @staticmethod
    def evaluate_f1(model, eval_set):
        model.eval()
        sents, labels = zip(*eval_set)
        with torch.no_grad():
            logits = model(sents, evall=True).cpu().numpy()
        model.train()
        preds = np.argmax(logits, axis=1)
        f1 = f1_score(labels, preds, average='weighted')
        return f1


def roberta_expand_data(data, tokenizer):
    sents, labels = zip(*data)
    tokens_masks = tokenizer(sents, padding=True)
    tokens = tokens_masks['input_ids']
    masks = tokens_masks['attention_mask']
    return tokens, masks, labels


def roberta_get_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return tokenizer


class RobertaTrainer():
    def __init__(self, model, device):
        self.model = model
        self.model.to(device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.device = device
        self.trained = False

    def train_batch(self, tokens, masks, labels):
        self.model.train()
        self.model.zero_grad()
        max_len = int(max(masks.sum(dim=1)).item())
        loss, _ = self.model(input_ids=tokens[:, :max_len], attention_mask=masks[:, :max_len], labels=labels, evall=False)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, train_data, ms_data, batchsize, max_epoch, patience, seed=None, verbose=False):
        if seed is None:
            seed = self.model.seed
        train_tokens, train_masks, train_labels = train_data
        train_tokens = torch.tensor(train_tokens).long().to(self.device)
        train_masks = torch.tensor(train_masks).float().to(self.device)
        train_labels = torch.tensor(train_labels).long().to(self.device)

        assert len(train_tokens) % batchsize == 0, 'train set size must be a multiple of batch size'
        n_batches = int(len(train_tokens) / batchsize)
        best_metric = float('-inf')
        cur_patience = 0
        torch.manual_seed(seed)
        for e in range(max_epoch):
            self.model.train()
            for b in range(n_batches):
                self.model.zero_grad()
                use_tokens = train_tokens[b * batchsize : (b + 1) * batchsize]
                use_masks = train_masks[b * batchsize : (b + 1) * batchsize]
                use_labels = train_labels[b * batchsize : (b + 1) * batchsize]
                self.train_batch(use_tokens, use_masks, use_labels)
            f1 = self.evaluate_f1(self.model, ms_data, batchsize)
            if f1 > best_metric:
                best_metric = f1
                cur_patience = 0
                best_model = copy(self.model)
            else:
                cur_patience += 1
            if cur_patience == patience:
                break
            if verbose:
                print(f'epoch {e}, f1: {f1:0.3f}')
        torch.manual_seed(random.randint(0, sys.maxsize))
        self.best_model = best_model
        return best_model, best_metric

    @staticmethod
    def evaluate_f1(model, data, batchsize):
        gpu_idx = next(model.parameters()).device
        tokens, masks, labels = data
        tokens = torch.tensor(tokens).long().to(gpu_idx)
        masks = torch.tensor(masks).float().to(gpu_idx)
        num_batches = math.ceil(len(tokens) / batchsize)
        all_pred_labels = []
        model.eval()
        with torch.no_grad():
            for b in range(num_batches):
                use_tokens = tokens[b * batchsize : (b + 1) * batchsize]
                use_masks = masks[b * batchsize : (b + 1) * batchsize]
                max_len = int(max(use_masks.sum(dim=1)).item())
                pred, = model(input_ids=use_tokens[:, :max_len], attention_mask=use_masks[:, :max_len], evall=True)
                pred_labels = pred.cpu().numpy().argmax(axis=1)
                all_pred_labels += list(pred_labels.flat)
        model.train()
        f1 = f1_score(labels, all_pred_labels, average='weighted')
        return f1

def get_trainer(model, model_seed, domain, device, mcdropout=False):
    assert model in ['cnn', 'lstm', 'aoe', 'roberta'], 'invalid model choice'
    if model != 'roberta':
        data = pickle.load(open('data/TOP.pkl', 'rb'))
        embeddings = data['GLOVE_EMBEDDING']
        vocabs = {v: i for i, v in enumerate(embeddings.keys())}
        embs = np.array(list(embeddings.values()))
        num_classes = int(len(data[domain]['intent_label_mapping']) / 2)
        def tokenize(sent):
            words = nltk.word_tokenize(sent)
            tks = list([vocabs.get(w, len(vocabs)) for w in words])
            return tks, words
        torch.manual_seed(model_seed)
        if model == 'lstm':
            model = BiLSTMModel(num_classes, embs, tokenize, mcdropout)
        elif model == 'cnn':
            model = CNNModel(num_classes, embs, tokenize, mcdropout)
        elif model == 'aoe':
            model = AOEModel(num_classes, embs, tokenize, mcdropout)
        torch.manual_seed(random.randint(0, sys.maxsize))
        trainer = Trainer(model, device)
        return trainer
    else:
        pretrained = roberta_get_model(model_seed, mcdropout)
        return RobertaTrainer(copy(pretrained), device)
