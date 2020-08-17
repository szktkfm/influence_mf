import pandas as pd
import numpy as np
import pickle


class AmazonDataset:


    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'

        self.load_train_test()
        self.load_user_items_dict()


    def load_train_test(self):
        self.user_item_train_df = pd.read_csv(self.data_dir + 'user_item_train.csv')
        self.user_item_test_df = pd.read_csv(self.data_dir + 'user_item_test.csv')
        self.user_item_train_nega_df = pd.read_csv(self.data_dir + 'user_item_train_nega.csv')

        self.user_list = []
        self.item_list = []
        with open(self.data_dir + 'user_list.txt', 'r') as f:
            for l in f:
                self.user_list.append(l.replace('\n', ''))

        with open(self.data_dir + 'item_list.txt', 'r') as f:
            for l in f:
                self.item_list.append(l.replace('\n', ''))

        # target
        y_train = [1 for i in range(len(self.user_item_train_df))] \
                        + [0 for i in range(len(self.user_item_train_nega_df))] 
        self.y_train = np.array(y_train)

                
    def load_user_items_dict(self):
        self.user_items_test_dict = pickle.load(open(self.data_dir + 'user_items_test_dict.pickle', 'rb'))
        #self.user_items_nega_dict = pickle.load(open(self.data_dir + 'user_items_nega_dict.pickle', 'rb'))
        
        
    def get_batch(self, batch_size=2):
        train_num = len(self.user_item_train_df) + len(self.user_item_train_nega_df)
        batch_idx = np.random.permutation(train_num)[:batch_size]
        # posi_tripletとnega_tripletを連結
        batch = pd.concat([self.user_item_train_df, self.user_item_train_nega_df]).values[batch_idx]
        batch_y_train = self.y_train[batch_idx]
    
        return batch, batch_y_train
    
    
    def get_nega_batch(self, users):
        nega_batch = []
        for user in users:
            nega_items = self.user_items_nega_dict[user]
            #print(nega_items)
        
            # ここ直す
            if len(nega_items) == 0:
                #nega_batch.append([user, item_list[np.random.randint(item_num)]])
                nega_batch.append([user, np.random.randint(len(self.item_list))])
                continue
        
            nega_item = nega_items[np.random.randint(len(nega_items))]
            nega_batch.append([user, nega_item])
    
        return np.array(nega_batch)
