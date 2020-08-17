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

def time_since(runtime):
    mi = int(runtime / 60)
    sec = runtime - mi * 60
    return (mi, sec)


def objective(trial):
    start = time.time()

    import gc
    gc.collect()
    
    data_dir = './data'

    dataset = dataloader.AmazonDataset(data_dir)
    embedding_dim = trial.suggest_discrete_uniform('embedding_dim', 16, 128, 16)
    user_size = len(dataset.user_list)
    item_size = len(dataset.item_list)
    layer_size = trial.suggest_int('layer_size', 1, 3)
    nfm = model.NFM(int(embedding_dim), user_size, item_size, layer_size).to(device)

    batch_size = int(trial.suggest_discrete_uniform('batch_size', 128, 512, 128))
    lr= trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    warmup = 350
    lr_decay_every = 2
    lr_decay_rate = trial.suggest_uniform('lr_decay_rate', 0.5, 1)

    iterater = training.TrainIterater(batch_size=batch_size, data_dir=data_dir)
    score = iterater.iterate_epoch(nfm, lr, epoch=1000, weight_decay=weight_decay, warmup=warmup, 
                            lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=10000)

    torch.cuda.empty_cache()

    mi, sec = time_since(time.time() - start)
    print('{}m{}sec'.format(mi, sec))

    return -1 * score


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=20)
    df = study.trials_dataframe() # pandasのDataFrame形式
    df.to_csv('./beauty_hyparams_result.csv')
    with open('beauty_best_param', 'wb') as f:
        pickle.dump(study.best_params, f)