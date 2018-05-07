import torch
import torch.utils.data as data
import numpy as np
import random


class User_Item_Dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_loader(data_path, train_negs = 4, test_negs = 99, batch_size = 100, num_workers = 2):
    # load file
    # each line contains (user_id, item_id, rating, timestamp)
    with open(data_path, 'r') as f:
        lines = f.readlines()

    # find user_max, item_max, rating_max
    max = [0, 0, 0]
    for i in range(len(lines)):
        lines[i] = lines[i].split("::")           # "::" for movielens , "," for csv file
        lines[i] = lines[i][:-1]  # remove timestamp
        lines[i] = [int(value) for value in lines[i]]
        lines[i][0] += - 1        # user_id: 1,2,3,4,... -> 0,1,2,3,...
        lines[i][1] += - 1        # item_id: 1,2,3,4,... -> 0,1,2,3,...
        #lines[i].append(1)        # dont use for movielens
        for j in range(3):
            if lines[i][j] > max[j]:
                max[j] = lines[i][j]
    max_user_id = max[0] + 1
    max_item_id = max[1] + 1
    max_rating = max[2]
    print("max_user_id : {}".format(max_user_id))
    print("max_item_id : {}".format(max_item_id))
    print("max_rating : {}".format(max_rating))

    # convert numpy array
    # divide data into train and test data
    lines = np.array(lines)
    test_index = np.zeros(max_user_id, dtype = np.int32)
    for i in range(len(lines)):
        user_id = lines[i, 0]
        if test_index[user_id] == 0:
            test_index[user_id] = i
    all_index = np.arange(len(lines))
    train_index = np.setdiff1d(all_index, test_index)
    train_lines = lines[train_index]

    # find negative map
    user_item_map = {}
    user_item_neg_map = {}
    for line in lines:
        user_id = line[0]
        item_id = line[1]
        if not user_id in user_item_map:
            user_set = set([item_id])
            user_item_map[user_id] = user_set
        else:
            user_item_map[user_id].add(item_id)
    for user_id in user_item_map:
        all_item_map = set(list(range(max_item_id)))
        user_item_neg_map[user_id] = all_item_map - user_item_map[user_id]

    # add negative samples to train data
    shape = train_lines.shape
    neg_data = np.zeros((shape[0] * train_negs, shape[1]), dtype = np.int32)
    for i, line in enumerate(train_lines):
        user_id = line[0]
        item_neg_map = user_item_neg_map[user_id]
        item_neg = random.sample(item_neg_map, train_negs)
        for j in range(train_negs):
            neg_data[i * train_negs + j] = [user_id, item_neg[j], 0]
    train_data = np.concatenate((train_lines, neg_data), axis = 0)

    # add negative samples to test data
    test_data = np.zeros((max_user_id, test_negs + 2), dtype = np.int32)
    test_data[:, :2] = lines[test_index][:, :2]
    for i in range(max_user_id):
        item_neg_map = user_item_neg_map[i]
        item_neg = random.sample(item_neg_map, test_negs)
        test_data[i, 2:] = item_neg

    # get infer data
    infer_length = max_user_id * max_item_id - len(lines)
    infer_data = np.zeros((infer_length, 2), dtype = np.int32)
    count = 0
    for user_id in user_item_neg_map:
        neg_item_list = list(user_item_neg_map[user_id])
        for neg_item in neg_item_list:
            infer_data[count] = [user_id, neg_item]
            count += 1

    # normalize
    train_data[:, 2] = train_data[:, 2] * 1.0 / max_rating

    # numpy to torch
    train_data, test_data, infer_data = \
        torch.LongTensor(train_data), torch.LongTensor(test_data), torch.LongTensor(infer_data)

    # get loader
    train_data = User_Item_Dataset(train_data)
    test_data = User_Item_Dataset(test_data)
    infer_data = User_Item_Dataset(infer_data)

    train_loader = data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    test_loader = data.DataLoader(test_data, batch_size = max_user_id, shuffle = True, num_workers = num_workers)
    infer_loader = data.DataLoader(infer_data, batch_size = batch_size, shuffle = False, num_workers = num_workers)

    print()
    print("Complete Data Processing!!")
    print()

    return max_user_id, max_item_id, train_loader, test_loader, infer_loader
