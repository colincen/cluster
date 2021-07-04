import torch
import torch.nn as nn
from datapreprocessing import get_dataloader
from config import get_params, init_experiment
from tqdm import tqdm
import numpy as np
from sfmodel import SlotFilling
import os
import json
import functools
from prettytable import PrettyTable
import csv


params = get_params()
logger = init_experiment(params, params.tgt_domain+"_"+ params.logger_filename)
dataloader_tgt, dataloader_tr, dataloader_val, dataloader_test, src_tag2idx, tgt_tag2idx = get_dataloader('AddToPlaylist',32,params.file_path,params.bert_path)
model = SlotFilling(params)