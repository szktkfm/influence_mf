import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

import dataloader
import evaluate
import model
import training

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_params():
    #return pickle.load(open('./result/best_param', 'rb'))
    return pickle.load(open('./beauty_best_param', 'rb'))


def time_since(runtime):
    mi = int(runtime / 60)
    sec = runtime - mi * 60
    return (mi, sec)


if __name__ == '__main__':
    data_dir = './data'
    params = load_params()

    dataset = dataloader.AmazonDataset(data_dir)

    embedding_dim = params['embedding_dim']
    user_size = len(dataset.user_list)
    item_size = len(dataset.item_list)
    layer_size = int(params['layer_size'])
    batch_size = int(params['batch_size'])
    lr = params['lr']
    weight_decay = params['weight_decay']
    warmup = 350
    lr_decay_every = 2
    lr_decay_rate = params['lr_decay_rate']

    nfm = model.NFM(int(embedding_dim), user_size, item_size, layer_size).to(device)
    iterater = training.TrainIterater(batch_size=batch_size, data_dir=data_dir)

    score = iterater.iterate_epoch(nfm, lr, epoch=1000, weight_decay=weight_decay, warmup=warmup, 
                            lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=100)

    torch.cuda.empty_cache()

    np.savetxt('score.txt', np.array([score]))