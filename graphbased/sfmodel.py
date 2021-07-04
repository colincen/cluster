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
        self.W = nn.Lin
        
            
