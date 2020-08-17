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


# 学習済みモデル読み込み
# loss_func定義

# 任意のテストデータ[u_i, i_i]ロード

# trainingデータのinfluenceを計算
## [u_i, i_i]のどちらかを含むtrainingデータを持ってくる
