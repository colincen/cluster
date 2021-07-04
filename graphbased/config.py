import argparse
import os
import subprocess
import pickle
import logging
import time
import random
from datetime import timedelta
import numpy as np
from tqdm import tqdm
import pickle
import numpy as np


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Cross-domain SLU")
    parser.add_argument("--exp_name", type=str, default="sf_model", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="cross-domain-slu.log")
    parser.add_argument("--dump_path", type=str, default="/home/shenhao/cluster/graphbased/", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")
    parser.add_argument("--log_file", type=str, default="/home/shenhao/cluster/graphbased/log")

    # adaptation parameters
    parser.add_argument("--epoch", type=int, default=20, help="number of maximum epoch")
    parser.add_argument("--tgt_domain", type=str, default="", help="target_domain")
    parser.add_argument("--bert_path", type=str, default="/home/shenhao/bert-base-uncased", help="embeddings file")  
    # slu_word_char_embs_with_slotembs.npy
    parser.add_argument("--file_path", type=str, default="/home/shenhao/data/coachdata/snips", help="embedding dimension") #400
    parser.add_argument("--emb_dim", type=int, default=768, help="embedding dimension") #400
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

   
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
 
    parser.add_argument("--domain", type=str, default="atp", help="dictionary type: slot embeddings based on each domain")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_saved_path", type=str, default="/home/shenhao/cluster/graphbased")

    params = parser.parse_args()

    return params

def init_experiment(params, logger_filename):
    """
    Initialize the experiment:
    - save parameters
    - create a logger
    """
    # save parameters
    get_saved_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)


    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def get_saved_path(params):
    """
    create a directory to store the experiment
    """
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    exp_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    if params.exp_id == "":
        chars = "0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(0, 3))
            if not os.path.isdir(os.path.join(exp_path, exp_id)):
                break
    else:
        exp_id = params.exp_id
    # update dump_path
    params.dump_path = os.path.join(exp_path, exp_id)
    print(params.dump_path)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
    assert os.path.isdir(params.dump_path)