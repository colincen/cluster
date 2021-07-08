import torch
import torch.nn as nn
from datapreprocessing import get_dataloader
from config import get_params, init_experiment
from tqdm import tqdm
import numpy as np
from sfmodel import SlotFilling
from conlleval import evaluate
from collections import Counter
import os
import json
import functools
from prettytable import PrettyTable
import csv
def save_model(params, src_tag2idx, logger, optimizer):
    model_saved_path = os.path.join(params.dump_path, "best_model.pth")
    
    torch.save({
            "model": model.state_dict(),
            "src_tag2idx":src_tag2idx,
        }, model_saved_path)
    logger.info("Best model has been saved to %s" % model_saved_path)

    opti_saved_path = os.path.join(params.dump_path, "opti.pth")
    torch.save(optimizer.state_dict(), opti_saved_path)
    logger.info("Best model opti has been saved to %s" % opti_saved_path)

def load_model(params):
    model_saved_path = os.path.join(params.dump_path, "best_model.pth")
    opti_saved_path = os.path.join(params.dump_path, "opti.pth")
    res_dict = torch.load(model_saved_path)
    model_params = res_dict['model']
    src_tag2idx = res_dict['src_tag2idx']
    model = SlotFilling(params, tag2idx=src_tag2idx, bert_path=params.bert_path, device=params.device)
    model.to(params.device)
    model.load_state_dict(model_params)
    return model
    



params = get_params()
params.batch_size = 64
logger = init_experiment(params, params.tgt_domain+"_"+ params.logger_filename)
dataloader_tgt, dataloader_tr, dataloader_val, dataloader_test, src_tag2idx, tgt_tag2idx = get_dataloader('AddToPlaylist',32,params.file_path,params.bert_path)
f = lambda d : {v:k for k,v in d.items()}
src_idx2tag = f(src_tag2idx)
tgt_idx2tag = f(tgt_tag2idx)

'''
model = SlotFilling(params, tag2idx=src_tag2idx, bert_path=params.bert_path, device=params.device)
optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
model.to(params.device)


best_f1 = 0
for epoch in range(params.epoch):
    model.train()
    pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
    total_loss_list = []
    for i, (words, x, is_heads, heads, tags, y, domains, seqlens) in pbar:
        y= y.to(params.device)
        x = x.to(params.device)
        heads=heads.to(params.device)

        loss = model(x=x, heads=heads,seq_len=seqlens,iseval=False,y=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_list.append(loss.item())
        pbar.set_description("(Epoch {})  Loss{:.4f}".format \
                ((epoch+1), \
                    np.mean(total_loss_list)
                    ))
    
    pbar = tqdm(enumerate(dataloader_val), total=len(dataloader_val))
    model.eval()
    _pred = []
    _gold = []
    for i, (words, x, is_heads, heads, tags, y, domains, seqlens) in pbar:
        x = x.to(params.device)
        heads = heads.to(params.device)
        res = model(x=x,heads=heads,seq_len=seqlens,iseval=True)
        
        for j in range(len(heads)):

            for k in range(1, seqlens[j]-1):
                if heads[j][k].item() == 1:
                    pred_tag = src_idx2tag[res[j][k].item()]
                    gold_tag = src_idx2tag[y[j][k].item()]
                    _pred.append(pred_tag)
                    _gold.append(gold_tag)
    
    (prec, rec, f1), d = evaluate(_pred, _gold, logger)
    if f1>best_f1:
        save_model(params,src_tag2idx,logger,optimizer)
'''


model = load_model(params)
model.eval()

pbar = tqdm(enumerate(dataloader_tgt), total=len(dataloader_tgt))
matrix = []
for i, (x, is_heads, heads, rep_idx, triple_pair, seqlens) in pbar:
    # print(x)
    # print(is_heads)
    # print(heads)
    # print(rep_idx)
    # print(triple_pair)
    # print(seqlens)
    x = x.to(params.device)
    heads = heads.to(params.device)
    res = model(x=x,heads=heads,seq_len=seqlens,iseval=True)

    utter = []
    for j in range(len(heads)):
        tempy = []
        yy = []
        for k in range(1, seqlens[j]-1):
            if heads[j][k].item() == 1:
                pred_tag = src_idx2tag[res[j][k].item()]
                tempy.append(pred_tag)
        for tok in tempy[rep_idx[j][0] : rep_idx[j][1]]:
            # print(tok)
            if tok != 'O':
                yy.append(tok[2:])
            else:
                yy.append('O')
        # print('-'*10)
        # print(triple_pair[1])
        # print(Counter(tempy).most_common(1)[0][0])
        matrix.append((triple_pair[j][0], triple_pair[j][1] ,Counter(yy).most_common(1)[0][0]))

fw = open('matrix.txt','w')
for line in matrix:
    fw.write(line[0] + "\t" + line[1]+"\t"+line[2]+"\n")




