import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

class SlotFilling(nn.Module):
    def __init__(self, params, tag2idx, bert_path, device):
        super(SlotFilling, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.encoder = BertModel.from_pretrained(bert_path)
        self.device = device
        self.params = params
        self.W = nn.Linear(768, int(len(tag2idx)))
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x, heads, seq_len, iseval = False, y = None):
        bsz, seq_len = x.size(0),x.size(1)
        attention_mask = (x != 0).byte().to(self.device)  
        reps = self.encoder(x, attention_mask=attention_mask)[0] 
        bert_out_reps = self.dropout(reps)
        logits = self.W(bert_out_reps)
        logits = self.dropout(logits)

        loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        if not iseval:
            loss = loss_func(logits.view(bsz*seq_len,-1), y.view(-1))
            return loss
        else:   
            return logits.argmax(-1)