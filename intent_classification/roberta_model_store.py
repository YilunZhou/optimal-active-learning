
import random
from copy import deepcopy as copy
import torch
from torch import nn
from transformers import RobertaForSequenceClassification

import logging
logging.basicConfig(level=logging.ERROR)

'''
A module to ensure that the RoBERTa model is loaded from disk only once.
'''

class RobertaWrapper(nn.Module):
    def __init__(self, model, seed, mcdropout):
        super(RobertaWrapper, self).__init__()
        self.model = model
        self.seed = seed
        self.dropout = model.classifier.dropout
        if mcdropout:
            self.dropout.p = 0.5
        self.mcdropout = mcdropout

    def forward(self, input_ids, attention_mask=None, labels=None, evall=None):
        assert evall is not None, 'evall must be specified'
        if self.mcdropout and not evall:
            self.dropout.train()
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

models = dict()

def roberta_get_model(seed, mcdropout):
    global models
    if (seed, mcdropout) in models:
        return copy(models[(seed, mcdropout)])
    torch.manual_seed(seed)
    base_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=7)
    torch.manual_seed(random.randint(0, 100000))
    model = RobertaWrapper(base_model, seed, mcdropout)
    models[(seed, mcdropout)] = model
    return copy(model)
