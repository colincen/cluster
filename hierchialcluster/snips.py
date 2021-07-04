import os
import random
import torch
from sklearn.cluster import KMeans
from transformers import BertModel, BertTokenizer
import torchtext
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import pandas as pd

datadir = '/home/shenhao/data/coachdata/snips/'
glovepath = '/home/shenhao/data'
charNgrampath = '/home/shenhao/data'
bertpath = '/home/shenhao/bert-base-uncased'

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather",\
     "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

sz=50

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
        slot2values[k] = slot2values[k][:sz]
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
    kmeans = KMeans(n_clusters=7, random_state= 0).fit(temp_embs)
    X = []
    Y = []
    i = 0
    for k in emb.keys():
        y = []
        for j in range(i,i + sz):
            y.append(kmeans.labels_[j])
        i = i + sz
        y = max(y, key=y.count)
        X.append(k)
        Y.append(y)
    coarse_dict = {}
    for i in range(len(Y)):
        if Y[i] not in coarse_dict:
            coarse_dict[Y[i]] = []
        coarse_dict[Y[i]].append(X[i])

    return coarse_dict

def hierarchycluster(slot2values, emb):
    fw1 = open('result1.txt', 'w')
    fw2 = open('result2.txt', 'w')
    df = pd.DataFrame(columns=('slot','1','2','3','4','5','6','7','8'))
    temp_embs = []
    for k,v in emb.items():
        temp_embs.append(v)


    temp_embs = np.concatenate(temp_embs, -1)
    temp_embs = temp_embs.transpose(1,0)

    mod_length = []
    for vec in temp_embs:
        mod_length.append(np.sqrt(np.dot(vec, vec)))
    # print(mod_length)

    print(temp_embs.shape)
    Z = sch.linkage(temp_embs, method = 'ward')
    y_ = sch.fcluster(Z, 45, 'distance')
    i = 0
    j = 0
    for k,v in emb.items():
        # print('++++++++++')
        # print(slot2values[k])
        print('-'*20)
        fw1.write(k+'\n')
        fw2.write(k+'\n')
        for val, lab in zip(slot2values[k], y_[i:i+sz]):
            # print("%s %d" % (val, lab))
            fw1.write(" ".join(val)+"\t"+str(lab)+'\t' +str(mod_length[j])+'\n')
            j += 1
        cnt_dict = dict(Counter(y_[i:i+sz]))
        s = pd.Series({'slot':k,
        '1':cnt_dict[1] if 1 in cnt_dict else 0,
        '2':cnt_dict[2] if 2 in cnt_dict else 0,
        '3':cnt_dict[3] if 3 in cnt_dict else 0,
        '4':cnt_dict[4] if 4 in cnt_dict else 0,
        '5':cnt_dict[5] if 5 in cnt_dict else 0,
        '6':cnt_dict[6] if 6 in cnt_dict else 0,
        '7':cnt_dict[7] if 7 in cnt_dict else 0,
        '8':cnt_dict[8] if 8 in cnt_dict else 0})
        df = df.append(s, ignore_index=True)
        i += 50
        fw1.write('\n')
        print('-'*20)
    dendrogram = sch.dendrogram(Z)
    

    plt.title("Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Euclidean distances")
    # plt.show()
    plt.savefig('pic1.png')
    df.to_csv('result2.csv')


data = get_data(datadir)
bert_reps, glove_reps, charN_reps = slot2emb(data)
hierarchycluster(data, bert_reps)

# coarse_dict = kmeans_cluster(bert_reps)
# print(coarse_dict)
# dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
# plt.title("Dendrogram")
# plt.xlabel("Customers")
# plt.ylabel("Euclidean distances")
# plt.show()
# plt.savefig('table.png')