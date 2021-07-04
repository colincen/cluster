import os
import random
import torch
from sklearn.cluster import KMeans
from transformers import BertModel, BertTokenizer
import torchtext
import numpy as np 


datadir = '/home/shenhao/data/coachdata/multiwoz/'
glovepath = '/home/shenhao/data'
charNgrampath = '/home/shenhao/data'
bertpath = '/home/shenhao/bert-base-uncased'

domain_set = ["train", "hotel", "attraction",\
     "taxi", "restaurant"]



def get_data(datadir):
    data = []
    slot2values = {}
    for domain in domain_set:
        p = os.path.join(datadir, domain, domain+'.txt')
        for line in open(p, 'r'):
            line = line.strip()
            toks, labels = line.split('\t')
            toks = toks.split()
            labels = labels.split()
            assert len(toks) == len(labels)
            data.append((toks, labels))
    
    for toks, labels in data:
        i = 0
        while i < len(labels):
            if labels[i] != 'O':
                j = i+1
                while j < len(labels) and labels[j][0] == 'I' \
                and labels[j][2:] == labels[i][2:]:
                    j += 1
                
                if labels[i][2:] not in slot2values:
                    slot2values[labels[i][2:]] = []

                slot2values[labels[i][2:]].append(toks[i:j])
                i = j
            else: i += 1

    for k in slot2values.keys():
        random.shuffle(slot2values[k])
        slot2values[k] = slot2values[k][:20]

    return slot2values

def slot2emb(data):
    Bertemb = {}
    Gloveemb = {}
    charNgramemb = {}
    tokenizer = BertTokenizer.from_pretrained(bertpath)
    encoder = BertModel.from_pretrained(bertpath)
    encoder = encoder.to('cuda:1')

    for k,v in data.items():
        for temp in v:
            toks = tokenizer.encode(' '.join(temp))
            toks = torch.tensor(toks, device='cuda:1')
            toks = toks.unsqueeze(0)
            reps = encoder(toks)[0].mean(1).squeeze()
            reps = reps.detach().cpu().numpy()
            if k not in Bertemb:
                Bertemb[k] = []
            Bertemb[k].append(np.reshape(reps, (768, 1)))
        Bertemb[k] = np.concatenate(Bertemb[k], -1)
    
    glove = torchtext.vocab.GloVe(cache=glovepath, name='6B')
    for k,v in data.items():
        for temp in v:
            slot_tokens = []
            for i in temp:
                slot_tokens.append(np.reshape(glove[i], (300,1)))
            slot_tokens = np.concatenate(slot_tokens, -1)
            reps = np.mean(slot_tokens,-1)
            if k not in Gloveemb:
                Gloveemb[k] = []
            Gloveemb[k].append(np.reshape(reps, (300, 1)))
        Gloveemb[k] = np.concatenate(Gloveemb[k], -1)
            
    char_ngram_model = torchtext.vocab.CharNGram(cache=charNgrampath)
    for k,v in data.items():
        for temp in v:
            slot_tokens = []
            for i in temp:
                slot_tokens.append(np.reshape(char_ngram_model[i], (100,1)))
            slot_tokens = np.concatenate(slot_tokens, -1)
            reps = np.mean(slot_tokens,-1)
            if k not in charNgramemb:
                charNgramemb[k] = []
            charNgramemb[k].append(np.reshape(reps, (100, 1)))
        charNgramemb[k] = np.concatenate(charNgramemb[k], -1)
    
    
    return Bertemb, Gloveemb, charNgramemb

    


def kmeans_cluster(emb):

    temp_embs = []
    for k,v in emb.items():
        temp_embs.append(v)

    temp_embs = np.concatenate(temp_embs, -1)
    temp_embs = temp_embs.transpose(1,0)
    print(temp_embs.shape)
    kmeans = KMeans(n_clusters=3, random_state= 0).fit(temp_embs)
    X = []
    Y = []
    i = 0
    for k in emb.keys():
        y = []
        for j in range(i,i +20):
            y.append(kmeans.labels_[j])
        i = i + 20
        y = max(y, key=y.count)
        X.append(k)
        Y.append(y)
    coarse_dict = {}
    for i in range(len(Y)):
        if Y[i] not in coarse_dict:
            coarse_dict[Y[i]] = []
        coarse_dict[Y[i]].append(X[i])

    return coarse_dict



data = get_data(datadir)
bert_reps, glove_reps, charN_reps = slot2emb(data)
coarse_dict = kmeans_cluster(bert_reps)
print(coarse_dict)
