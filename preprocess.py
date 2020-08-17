import pandas as pd
import numpy as np
import pickle


# negative sampling
def negative_sampling(train_df, user_list, item_list):
    # implicit_feed = [list(r) for r in user_item_df.values]
    implicit_feed = [list(r) for r in train_df.values]

    user_item_train_nega = []

    count = 0
    #while count < 1000:
    while count < len(train_df):
        #user = user_list[np.random.randint(user_num)]
        #item = item_list[np.random.randint(item_num)]
        user = np.random.randint(len(user_list))
        item = np.random.randint(len(item_list))
        if [user, item] in implicit_feed:
            continue
        if [user, item] in user_item_train_nega:
            continue

        user_item_train_nega.append([user, item])
        count += 1

    user_item_train_nega_df = pd.DataFrame(user_item_train_nega, columns=['reviewerID', 'asin'])

    return user_item_train_nega_df


# データに含まれるuser-item1, item2, item3, ...を返す
# 辞書
def user_aggregate_item(user_list, df):
    user_items_dict = {}
    #for user in user_list:
    for i in range(len(user_list)):
        items_df = df[df['reviewerID'] == i]
        user_items_dict[i] = list(items_df['asin'])
    return user_items_dict



if __name__ == '__main__':
    # データ読み込み
    user_item_df = pd.read_csv('./user_item.csv')
    item_list = list(set(list(user_item_df['asin'])))
    user_list = list(set(list(user_item_df['reviewerID'])))
    print('item size: {}'.format(len(item_list)))
    print('user size: {}'.format(len(user_list)))
    # 保存
    with open('./data/user_list.txt', 'w') as f:
        for user in user_list:
            f.write(user + '\n')
    #np.savetxt('user_list.txt', np.array(user_list))
    with open('./data/item_list.txt', 'w') as f:
        for item in item_list:
            f.write(item + '\n')


    # user_itemをID化
    user_item_list = []
    for row in user_item_df.values:
        user = user_list.index(row[0])
        item = item_list.index(row[1])
        user_item_list.append([user, item])

    user_item_df = pd.DataFrame(np.array(user_item_list),
                                columns = ['reviewerID', 'asin'])

    # train-testスプリット
    user_item_df = user_item_df.take(np.random.permutation(len(user_item_df)))
    train_num = int(0.5 * len(user_item_df))
    user_item_train_df = user_item_df[0:train_num]
    user_item_test_df = user_item_df[train_num:]

    print('train {}'.format(train_num))
    print('test {}'.format(len(user_item_test_df)))
    # スプリットを保存
    user_item_train_df.to_csv('./data/user_item_train.csv', index=False)
    user_item_test_df.to_csv('./data/user_item_test.csv', index=False)

    # negative sampling
    user_item_train_nega_df = negative_sampling(user_item_train_df, user_list, item_list)
    # negative sampleを保存
    user_item_train_nega_df.to_csv('./data/user_item_train_nega.csv', index=False)

    # データに含まれるuser-item1, item2, item3, ...
    #user_items_nega_dict = user_aggregate_item(user_list, user_item_train_nega_df)
    user_items_test_dict = user_aggregate_item(user_list, user_item_test_df)
    # user_items_dictを保存
    #with open('./data/user_items_nega_dict.pickle', 'wb') as f:
    #    pickle.dump(user_items_nega_dict, f)
    with open('./data/user_items_test_dict.pickle', 'wb') as f:
        pickle.dump(user_items_test_dict, f)
