
import math, pickle, random, sys
from copy import deepcopy as copy

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, embedding_data, vocab):
        '''
        embedding_data is an np.array of shape num_vocab x emb_dim
        vocab is a dictionary from word to token_id in embedding_data
        '''
        super(Encoder, self).__init__()
        num_embedding, emb_dim = embedding_data.shape
        self.unk_idx = num_embedding
        self.sos_idx = num_embedding + 1
        self.eos_idx = num_embedding + 2
        self.pad_idx = num_embedding + 3
        embedding_data = torch.cat((torch.tensor(embedding_data).float(),
                                         torch.randn(1, emb_dim) * 0.01,             # <unk>
                                         torch.randn(1, emb_dim) * 0.01,             # <sos>
                                         torch.randn(1, emb_dim) * 0.01,             # <eos>
                                         torch.zeros(1, emb_dim)                     # <pad>
                                    ), dim=0)
        self.embedding = nn.Embedding.from_pretrained(embedding_data, freeze=False,
                                                      padding_idx=self.pad_idx)
        self.vocab = vocab
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim,
                            batch_first=True, bidirectional=True)
        self.repr_dim = 2 * emb_dim

    def forward(self, sentences):
        '''
        run the encoder on the batch of sentences
        sentences is a list of tokenized word lists
        '''
        tokens_list = [[self.vocab.get(w, self.unk_idx) for w in s] for s in sentences]
        lens = list(map(len, tokens_list))
        max_len = max(lens)
        tokens_array = np.ones((len(sentences), max_len + 2)) * self.pad_idx
        for i, ts in enumerate(tokens_list):
            tokens_array[i, 0] = self.sos_idx
            tokens_array[i, 1:len(ts) + 1] = ts
            tokens_array[i, len(ts) + 1] = self.eos_idx
        tokens_tensor = torch.tensor(tokens_array).long().to(self.get_device())
        embedding_tensor = self.embedding(tokens_tensor)
        embeddings_pack = pack_padded_sequence(embedding_tensor, torch.tensor(lens) + 2,
                                               batch_first=True, enforce_sorted=False)
        lstm_hidden_pack, (_, _) = self.lstm(embeddings_pack)
        lstm_hidden, unpacked_lens = pad_packed_sequence(lstm_hidden_pack, batch_first=True)
        return lstm_hidden

    def get_device(self):
        return next(self.parameters()).device

class Decoder(nn.Module):
    def __init__(self, num_labels, enc_repr_dim, self_hidden_size=50):
        super(Decoder, self).__init__()
        self.num_labels = num_labels
        self.enc_repr_dim = enc_repr_dim
        self.self_hidden_size = self_hidden_size
        self.init_project = nn.Linear(enc_repr_dim, self_hidden_size)
        self.lstm_cell = nn.LSTMCell(input_size=num_labels + 1 + enc_repr_dim,
                            hidden_size=self_hidden_size)
        self.out_project = nn.Linear(self_hidden_size, num_labels)
        self.go_idx = num_labels

    def forward(self, enc_repr, labels):
        '''
        forward in teacher forcing mode
        enc_repr is of shape (batchsize x (max_sent_len + 2) x enc_repr_dim)
        labels is a list of label tags
        '''
        lens = list(map(len, labels))
        batchsize = enc_repr.shape[0]
        max_len = max(lens)
        assert enc_repr.shape == (len(labels), max_len + 2, self.enc_repr_dim)
        init_hidden = enc_repr[range(batchsize), tuple([l + 1 for l in lens]), :]
        all_hiddens = []
        hidden = self.init_project(init_hidden)
        cell = torch.zeros((batchsize, self.self_hidden_size)).to(self.get_device())
        labels_array = np.ones((batchsize, max_len)) * self.go_idx
        for i, l in enumerate(labels):
            labels_array[i, :len(l)] = l
        labels_tensor = torch.tensor(labels_array).long().to(self.get_device())
        for i in range(max_len):
            if i == 0:  # use go_idx as the "previous token" in the first step
                forced_tk = torch.ones(batchsize).long() * self.go_idx
            else:  # use the previous teacher-forced token
                forced_tk = labels_tensor[:, i - 1]
            forced_tk_one_hot = torch.zeros(batchsize, self.num_labels + 1).to(self.get_device())
            forced_tk_one_hot[range(batchsize), forced_tk] = 1
            cur_enc_repr = enc_repr[:, i + 1, :]
            inpu = torch.cat((forced_tk_one_hot, cur_enc_repr), dim=1)
            new_hidden, new_cell = self.lstm_cell(inpu, (hidden, cell))
            all_hiddens.append(new_hidden)
            hidden = new_hidden
            cell = new_cell
        all_hiddens = torch.stack(all_hiddens, dim=1)
        all_hiddens = all_hiddens.view(-1, self.self_hidden_size)
        output = self.out_project(all_hiddens)
        valid_flag = labels_tensor.view(-1) != self.go_idx
        return output, labels_tensor.view(-1), valid_flag

    def tag(self, enc_repr, lens):
        '''forward in free generation mode by greedy decoding'''
        max_len = max(lens)
        assert enc_repr.shape[1:] == (max_len + 2, self.enc_repr_dim)
        batchsize = enc_repr.shape[0]
        init_hidden = enc_repr[range(batchsize), tuple([l + 1 for l in lens]), :]
        hidden = self.init_project(init_hidden)
        cell = torch.zeros((batchsize, self.self_hidden_size)).to(self.get_device())
        pred_logits = []
        pred_tks = []
        for i in range(max_len):
            if i == 0:  # use go_idx as the "previous token" in the first step
                prev_tk = torch.ones(batchsize).long() * self.go_idx
            prev_tk_onehot = torch.zeros(batchsize, self.num_labels + 1).to(self.get_device())
            prev_tk_onehot[range(batchsize), prev_tk] = 1
            cur_enc_repr = enc_repr[:, i + 1, :]
            inpu = torch.cat((prev_tk_onehot, cur_enc_repr), dim=1)
            new_hidden, new_cell = self.lstm_cell(inpu, (hidden, cell))
            hidden = new_hidden
            cell = new_cell
            pred = self.out_project(hidden)
            prev_tk = pred.argmax(dim=1)
            pred_logits.append(pred)
            pred_tks.append(prev_tk)
        return torch.stack(pred_logits, dim=1), torch.stack(pred_tks, dim=1)

    def get_device(self):
        return next(self.parameters()).device

class EncoderDecoder(nn.Module):
    def __init__(self, embedding_data, vocab, num_labels, dec_hidden_size=50):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(embedding_data, vocab)
        self.decoder = Decoder(num_labels, self.encoder.repr_dim, dec_hidden_size)

    def forward(self, sentences, labels):
        enc_repr = self.encoder(sentences)
        pred, labels_tensor, valid_flag = self.decoder(enc_repr, labels)
        return pred, labels_tensor, valid_flag

    def tag(self, sentences):
        lens = [len(s) for s in sentences]
        enc_repr = self.encoder(sentences)
        pred_logits, pred_tks = self.decoder.tag(enc_repr, lens)
        return pred_logits, pred_tks

class Trainer():
    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.model.to(device)
        self.loss = nn.CrossEntropyLoss(ignore_index=model.decoder.go_idx, reduction='sum')
        self.optimizer = Adam(self.model.parameters())
        self.trained = False

    def train_batch(self, batch):
        self.model.train()
        self.model.zero_grad()
        sents, tags = zip(*batch)
        logits_tensor, labels_tensor, _ = self.model(sents, tags)
        loss_val = self.loss(logits_tensor, labels_tensor)
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()

    def train(self, train_set, valid_set, batchsize, max_epoch, patience, verbose=False):
        '''
        train the model for max_epoch, and validate after every epoch.
        if the validation metric is not improving for patience epochs, early-terminate the training
        and return the best model
        patience == 1 means terminating the training as soon as the validation metric stops improving
        valid_func(model, valid_set) returns the validation metric (e.g. accuracy or loss) of the model
        a higher validation metric value is better
        '''
        assert not self.trained, 'The model in the trainer has been trained'
        self.trained = True
        # assert len(train_set) % batchsize == 0, 'train_set size must be a multiple of batch size'
        n_batches = math.ceil(len(train_set) / batchsize)
        best_valid_metric = float('-inf')
        cur_patience = 0
        for epoch in range(max_epoch):
            losses = []
            for i in range(n_batches):
                cur_batch = train_set[i * batchsize : (i + 1) * batchsize]
                loss = self.train_batch(cur_batch)
                losses.append(loss)
            cur_patience += 1
            valid_metric = self.evaluate_f1(self.model, *zip(*valid_set))
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                best_model = copy(self.model)
                cur_patience = 0
            elif cur_patience == patience:
                break
            if verbose:
                print(f'epoch {epoch}: loss {np.mean(losses):0.3f}, val metric {valid_metric:0.3f}')
        best_model.encoder.lstm.flatten_parameters()
        self.best_model = best_model
        return best_model, best_valid_metric

    @staticmethod
    def evaluate_f1(model, sents, tags):
        model.eval()
        lens = [len(l) for l in sents]
        with torch.no_grad():
            _, pred_tags = model.tag(sents)
        pred_tags = pred_tags.cpu().numpy()
        pred_tags = [list(ls[:l].flat) for ls, l in zip(pred_tags, lens)]
        pred_tags = sum(pred_tags, [])
        tags = sum(map(list, tags), [])
        f1 = f1_score(tags, pred_tags, average='weighted')
        return f1

def get_trainer(model_seed, device='cuda:0'):
    data = pickle.load(open('data/restaurant.pkl', 'rb'))
    glove = data['GLOVE_EMBEDDING']
    vocab_list = list(glove.keys())
    vocab = {v: i for i, v in enumerate(vocab_list)}
    embedding_data = np.array([glove[v] for v in vocab_list])
    num_labels = int(len(data['tag_mapping']) / 2)
    if model_seed >= 0:
        torch.manual_seed(model_seed)
    model = EncoderDecoder(embedding_data, vocab, num_labels)
    torch.manual_seed(random.randint(0, sys.maxsize))
    trainer = Trainer(model, device)
    return trainer
