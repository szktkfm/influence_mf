import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from importlib import reload

import dataloader
import evaluate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainIterater():


    def __init__(self, batch_size, data_dir):
        self.data_dir = data_dir
        self.dataset = dataloader.AmazonDataset(data_dir)
        self.batch_size = batch_size
        
        
    def train(self, batch, loss_func, optimizer, model):
        optimizer.zero_grad()

        batch, y_train = batch
        user_tensor = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
        item_tensor = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float, device=device)

        pred = model(user_tensor, item_tensor)
        #print(pred)
        loss = loss_func(pred, y_train)
        loss.backward()
        #print(loss)
        optimizer.step()

        return loss


    def iterate_train(self, model, lr=0.001, optimizer='Adam', 
                      weight_decay=0, print_every=2000, plot_every=50):
        # define optim
        if optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        # define loss_func
        # BPRの場合targetを全て1にする
        loss_func = nn.BCELoss()

        print_loss_total = 0
        plot_loss_list = []
        plot_loss_total = 0

        train_num = len(self.dataset.user_item_train_df)
        start_time = time.time()
        for i in range(int(train_num / self.batch_size) + 1):
            #batch = get_batch()
            batch = self.dataset.get_batch(batch_size=self.batch_size)

            loss = self.train(batch, loss_func, optimizer, model)

            print_loss_total += loss.detach()
            plot_loss_total += loss.detach()


            # print_everyごとに現在の平均のlossと、時間、dataset全体に対する進捗(%)を出力
            if (i+1) % print_every == 0:
                runtime = time.time() - start_time
                mi, sec = self.time_since(runtime)
                avg_loss = print_loss_total / print_every
                data_percent = int(i * self.batch_size / train_num * 100)
                print('train loss: {:e}    processed: {}({}%)    {}m{}sec'.format(
                    avg_loss, i*self.batch_size, data_percent, mi, sec))
                print_loss_total = 0

            # plot_everyごとplot用のlossをリストに記録しておく
            if (i+1) % plot_every == 0:
                avg_loss = plot_loss_total / plot_every
                plot_loss_list.append(avg_loss)
                plot_loss_total = 0
            
        return plot_loss_list

                
    def iterate_epoch(self, model, lr, epoch, optimizer='Adam', weight_decay=0, 
                      warmup=0, lr_decay_rate=1, lr_decay_every=10, eval_every=5):
        eval_model = evaluate.Evaluater(self.data_dir)
        plot_loss_list = []
        plot_score_list = []
                          
        for i in range(epoch):
            plot_loss_list.extend(self.iterate_train(model, lr=lr, optimizer=optimizer, 
                                                     weight_decay=weight_decay, print_every=500))
            
            # lrスケジューリング
            if i > warmup:
                if (i - warmup) % lr_decay_every == 0:
                    lr = lr * lr_decay_rate
                    
            if (i+1) % eval_every == 0:
                #score = eval_model.topn_precision(model)
                score = eval_model.topn_map(model)
                plot_score_list.append(score)
                #print('epoch: {}  precision: {}'.format(i, score))
                print('epoch: {}  map: {}'.format(i, score))
        
        #self._plot(plot_loss_list)
        #self._plot(plot_score_list)
        
        # とりあえず最後のepochのscoreを返す
        return eval_model.topn_map(model)
        
        
        
    def _plot(self, loss_list):
        # ここもっとちゃんと書く
        plt.plot(loss_list)
        plt.show()
         
        
    def time_since(self, runtime):
        mi = int(runtime / 60)
        sec = int(runtime - mi * 60)
        return (mi, sec)
