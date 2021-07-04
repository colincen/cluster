import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import random
slot_list = ['<PAD>','playlist', 'music_item', 'geographic_poi', 'facility', 
'movie_name', 'location_name', 'restaurant_name', 'track', 'restaurant_type', 
'object_part_of_series_type', 'country', 'service', 'poi', 'party_size_description',
'served_dish', 'genre', 'current_location', 'object_select', 'album', 'object_name',
'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 'artist', 
'cuisine', 'entity_name', 'object_type', 'playlist_owner', 'timeRange', 'city',
'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number',
'condition_description', 'condition_temperature']

y_set = ['<PAD>' ,'O', 'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 
'I-geographic_poi', 'B-facility', 'I-facility', 'B-movie_name', 'I-movie_name', 'B-location_name', 'I-location_name', 
'B-restaurant_name', 'I-restaurant_name', 'B-track', 'I-track', 'B-restaurant_type', 'I-restaurant_type', 
'B-object_part_of_series_type', 'I-object_part_of_series_type', 'B-country', 'I-country', 'B-service', 'I-service',
 'B-poi', 'I-poi', 'B-party_size_description', 'I-party_size_description', 'B-served_dish', 'I-served_dish', 
 'B-genre',  'I-genre', 'B-current_location', 'I-current_location', 'B-object_select', 'I-object_select', 
 'B-album', 'I-album', 'B-object_name', 'I-object_name', 'B-state', 'I-state', 'B-sort', 'I-sort',
  'B-object_location_type', 'I-object_location_type', 'B-movie_type', 'I-movie_type', 'B-spatial_relation', 'I-spatial_relation',
   'B-artist', 'I-artist', 'B-cuisine', 'I-cuisine', 'B-entity_name', 'I-entity_name', 'B-object_type', 'I-object_type', 
   'B-playlist_owner', 'I-playlist_owner', 'B-timeRange', 'I-timeRange', 'B-city', 'I-city', 'B-rating_value',
    'B-best_rating', 'B-rating_unit', 'B-year', 'B-party_size_number', 'B-condition_description', 'B-condition_temperature']

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather",\
     "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 
    'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}


class NerDataset(Dataset):
    def __init__(self, raw_data, tag2idx ,bert_path):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        sents, tags, domains = [], [], []
        for entry in raw_data:
            sents.append(["[CLS]"] + entry[0] + ["[SEP]"])
            tags.append(["<PAD>"] + entry[1] + ["<PAD>"])
            domains.append([entry[-1]])
        self.sents, self.tags, self.domains, self.tag2idx = sents, tags, domains, tag2idx
    
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        tag2idx = self.tag2idx
        words, tags, domains = self.sents[idx], self.tags[idx], self.domains[idx]



        domains = domain_set.index(domains[0])

        x, y = [], []
        is_heads = []
        for w, t in zip(words, tags):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)
            is_head = [1] + [0] * (len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)

            yy = [tag2idx[each] for each in t]

            

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
            

        assert len(x) == len(y) == len(is_heads)



        seq_len = len(y)
        

        words = " ".join(words)
        tags = " ".join(tags)

        return words, x, is_heads, tags, y, domains, seq_len

def read_file(fpath):
    raw_data = {}
    for intent in domain_set:
        raw_data[intent] = []
        for i, line in enumerate(open(fpath+'/'+intent+'/'+intent+'.txt')):
            temp = []
            tokens, labels = line.strip().split('\t')
            tokens = tokens.split()
            label_list = labels.split()
            if '������' in tokens:
                continue
            temp.append(tokens)
            temp.append(label_list)
            temp.append(intent)
            raw_data[intent].append(temp)
    
    return raw_data

def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(6)
    domains = f(5)
    maxlen = np.array(seqlens).max()


    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(4, maxlen)
    heads = f(2 ,maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, f(heads), tags, f(y), domains, seqlens, 

class tgtDataset(NerDataset):
    def __init__(self, raw_data, bert_path):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        sents, replaced_indexs, tuple_pairs = [],[],[]
        for entry in raw_data:
            sents.append(["[CLS]"] + entry[0] + ["[SEP]"])
            replaced_indexs.append(entry[1])
            tuple_pairs.append(entry[2])
        self.sents, self.replaced_indexs, self.tuple_pairs = sents, replaced_indexs, tuple_pairs
    
    def __getitem__(self, idx):
        words, rep_idx, triple_pair = self.sents[idx], self.replaced_indexs[idx], self.tuple_pairs[idx]
        x = []
        is_heads = []
        for w in words:
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)
            is_head = [1] + [0] *(len(tokens) - 1)
            x.extend(xx)
            is_heads.extend(is_head)
        
        assert len(x) == len(is_heads)
        seq_len = len(x)

        return x, is_heads, rep_index, triple_pair, seq_len

def pad_tgt(batch):
    f = lambda x : [sample[x] for sample in batch]
    is_heads = f(1)
    rep_idx = f(2)
    triple_pair = f(3)
    seqlens = f(4)
    max_len = np.array(seqlens).max()

    f = lambda x, seq_len: [sample[x] + [0] * (seq_len - len(sample[x])) for sample in batch]
    x = f(0, max_len)
    heads = f(1, max_len)

    f = torch.LongTensor

    return f(x), is_heads, f(heads), rep_idx, triple_pair, seqlens
   
def get_dataloader(tgt_domain, batch_size, fpath, bert_path):
    raw_data = read_file(fpath)
    src_tag2idx = {"<PAD>" : 0, "O":1}
    tgt_tag2idx = {"<PAD>" : 0, "O":1}
    src_train_data = []
    src_dev_data = []
    src_test_data = []
    tgt_data = []
    src_data = []


    for k, v in domain2slot.items():
        if k == tgt_domain:
            tgt_data.extend(raw_data[k])
            for slot in v:
                _B = "B-" + slot
                if _B not in tgt_tag2idx.keys() and _B in y_set:
                    tgt_tag2idx[_B] = len(tgt_tag2idx)
                _I = "I-" + slot
                if _I not in tgt_tag2idx.keys() and _I in y_set:
                    tgt_tag2idx[_I] = len(tgt_tag2idx)
        else:
            src_data.extend(raw_data[k])
            for slot in v:
                _B = "B-" + slot
                if _B not in src_tag2idx.keys() and _B in y_set:
                    src_tag2idx[_B] = len(src_tag2idx)
                _I = "I-" + slot
                if _I not in src_tag2idx.keys() and _I in y_set:
                    src_tag2idx[_I] = len(src_tag2idx)
    

    random.shuffle(src_data)

    src_train_data = src_data[:-2000]
    src_dev_data = src_data[-2000:-1500]
    src_test_data = src_data[-1500:]

    src_test_data_res = make_sents_to_pieces(src_test_data)
    tgt_data_res = make_sents_to_pieces(src_test_data + tgt_data)
    slot_value_pair = sample_slot_value_pair(src_test_data_res + tgt_data_res)
    expand_data = expand_test_data(src_test_data_res, slot_value_pair, 10)



    dataset_tr = NerDataset(raw_data=src_train_data, tag2idx=src_tag2idx, bert_path=bert_path)
    dataset_val = NerDataset(raw_data=src_dev_data, tag2idx=src_tag2idx, bert_path=bert_path)    
    dataset_test = NerDataset(raw_data=src_test_data, tag2idx=src_tag2idx, bert_path=bert_path)    

    dataset_tgt = tgtDataset(raw_data=expand_data, bert_path=bert_path)

   
    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=pad)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=pad)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=pad)

    dataloader_tgt = DataLoader(dataset=dataset_tgt, batch_size=batch_size, shuffle=False, collate_fn=pad_tgt)


    return dataloader_tgt, dataloader_tr, dataloader_val, dataloader_test, src_tag2idx, tgt_tag2idx

def make_sents_to_pieces(data):
    #[[x],[y],[domain]]
    res_data = []
    # [[sent], [label],[slot_names],[slot_values],[slot_indexs]]
    for line in data:
        example = [[],[],[],[],[]]
        sent, label, domain = line[0], line[1], line[2]
        example[0] = sent
        example[1] = label
        i = 0
        while i < len(sent):
            if label[i][0] == 'B':
                j = i+1
                while j < len(label) and label[i][2:] == label[j][2:]:
                    j += 1
                example[2].append(label[i][2:])
                example[3].append(sent[i:j])
                example[4].append((i,j))
                i = j
            i += 1
        res_data.append(example)
    
    return res_data                
                
def sample_slot_value_pair(data):
    slot2value = {}

    for line in data:
    # [[sent], [label],[slot_name],[slot_value],[slot_index]]
        for j in range(len(line[2])):
            slot_name, slot_value = line[2][j], line[3][j]
            if slot_name not in slot2value:
                slot2value[slot_name] = []
            slot2value[slot_name].append(slot_value)
    
    for k,v in slot2value.items():
        random.shuffle(slot2value[k])
    return slot2value

def expand_test_data(data, slot_value_pair, expand_num):
    # [[sent], [label],[slot_names],[slot_values],[slot_indexs]]
    # slot_value_pair {slot_name : []}
    res_data = []
    # import time
    # start_time = time.time()
    for slot_name, slot_values in slot_value_pair.items():
        for slot_value in slot_values[:expand_num]:
            for line in data:
                example = [None, None, None]
                #[sent,  replaced_span_index , [raw_slot, to_replace_slot, pred_slot]]
                span_idx = random.randint(0, len(line[4])-1)
                l,r = line[4][span_idx]
                # print(line[0])
                # print((l,r))
                example[0] = line[0][:l] + slot_value + line[0][r:]
                # print(example[0])
                new_l, new_r = l, l+len(slot_value)
                # print((new_l, new_r))
                # print('-'*20)
                example[1] = (new_l, new_r)
                example[2] = [line[2][span_idx], slot_name, None]
                res_data.append(example)
    # end_time = time.time()
    # print(end_time-start_time)
    return res_data


dataloader_tgt, dataloader_tr, dataloader_val, dataloader_test, src_tag2idx, tgt_tag2idx = get_dataloader('AddToPlaylist',32,'/home/shenhao/data/coachdata/snips','/home/shenhao/bert-base-uncased')
# make_sents_to_pieces(tgt_data)