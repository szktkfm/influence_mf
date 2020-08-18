import pickle
import pandas as pd
import numpy as np
import time

import dataloader
import evaluate
import model
import training
from influence import get_influence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna

data_dir = './data'
dataset = dataloader.AmazonDataset(data_dir)
evaluater = evaluate.Evaluater(data_dir)


# 学習済みモデル読み込み
mf = torch.load('model.torch')
# loss_func定義
loss_func = nn.BCELoss()

# 任意のテストデータ[u_i, i_i]ロード
# あるユーザに対してランキング上位のアイテム
target_user = dataset.user_list[0]
target_user = dataset.user_list.index(target_user)
ranking_idx = evaluater.predict(mf, target_user)
target_item = ranking_idx[0]


# trainingデータのinfluenceを計算
## [u_i, i_i]のどちらかを含むtrainingデータを持ってくる
train_data = []
for row in dataset.user_item_train_df.values:
    if target_user == row[0] or target_item == row[1]:
        train_data.append(row)

target_data = [target_user, target_item]
influ = get_influence(loss_func, train_data[0], mf)