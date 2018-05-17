import torch
import torch.utils.data as data
import numpy as np
import random


##### STEPS #####
# 1. load files & get max length of each column: user, item, rating
#
# 2. convert original file into np array & split it into train and test
#       -> lines, train_lines, train_index, test_index, all_index
#
# 3. make user-item map and negative map
#       -> user_item_map, user_item_neg_map
#
# 4. add negative samples to train and test data
#       -> train_data(len(train_lines)*5, 3):
#               [[usr id, item id, pos rating],
#               [usr id, item id, neg rating], ,,, ]
#       -> test_data(max_user_id, 2 + 99): [usr id, pos item id, neg item ids]
#
# 5. make infer data
#       -> infer_data(unrated lines, 2): [[usr id, neg item id], ,,, ]
#
# 6. normalize & np array to torch tensor & dataloader
#       -> train_loader, test_loader, infer_loader


class User_Item_Dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_loader(data_path, train_negs = 4, test_negs = 99, batch_size = 100, num_workers = 2):
    ########################################
    ### load file
    # each line contains (user_id, item_id, rating, timestamp)
    with open(data_path, 'r') as f:
        lines = f.readlines()

    ### find user_max, item_max, rating_max
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

    max_user_id = max[0] + 1 # cuz we substracted 1 from usr_id and item_id
    max_item_id = max[1] + 1
    max_rating = max[2]
    print("max_user_id : {}".format(max_user_id))
    print("max_item_id : {}".format(max_item_id))
    print("max_rating : {}".format(max_rating))


    ########################################
    ### convert numpy array
    # divide data into train and test data
    lines = np.array(lines)
    # lines.shape: (changeable) (1000209, 3)

    test_index = np.zeros(max_user_id, dtype = np.int32)
    # test_index: where the user_id first appears in lines
    for i in range(len(lines)):
        user_id = lines[i, 0]
        if test_index[user_id] == 0:
            test_index[user_id] = i

    all_index = np.arange(len(lines))

    train_index = np.setdiff1d(all_index, test_index)
    # train_index: sorted elements that are in test_index and not in all_index.
    train_lines = lines[train_index]


    ########################################
    ### find negative map: user_item_neg_map
    user_item_map = {}
    user_item_neg_map = {}
    for line in lines:
        user_id = line[0]
        item_id = line[1]
        if not user_id in user_item_map:
            ## when a certain user_id first appears,
            # make a set containing item_id and append it to map
            user_set = set([item_id])
            user_item_map[user_id] = user_set

        else:
            ## if the user_id is already in map,
            # add the item_id to the user's user_set
            user_item_map[user_id].add(item_id)
    # user_item_map: {0:{0,2691,1245,2686},1:{1356,3452,743}, ,,, }

    for user_id in user_item_map:
        all_item_map = set(list(range(max_item_id)))
        # all_item_map: {0,1,2, ,,, max_item_id}
        user_item_neg_map[user_id] = all_item_map - user_item_map[user_id]
        # user_item_neg_map: items_id that users didn't answer


    ########################################
    ### add negative samples to train data
    shape = train_lines.shape
    # shape.shape: (994169, 3)

    neg_data = np.zeros((shape[0] * train_negs, shape[1]), dtype = np.int32)
    # neg_data.shape: (994169 * 4, 3), all zeros

    for i, line in enumerate(train_lines):
        user_id = line[0]
        item_neg_map = user_item_neg_map[user_id]

        item_neg = random.sample(item_neg_map, train_negs)
        # item_neg: choose train_negs(default 4) neg items from each user

        for j in range(train_negs):
            neg_data[i * train_negs + j] = [user_id, item_neg[j], 0]
            # Ex) i=0, j=0: nd[0*4 + 0] = [0, random[0], 0]
            # Ex) i=0, j=1: nd[0*4 + 1] = [0, random[1], 0]
            # Ex) i=1, j=0: nd[1*4 + 0] = [0, random[0], 0]

    train_data = np.concatenate((train_lines, neg_data), axis = 0)
    # train_data.shape: (994169 * 5, 3)


    ### add negative samples to test data
    test_data = np.zeros((max_user_id, test_negs + 2), dtype = np.int32)
    # test_data.shape: (max_user_id, 2 + 99)
    # test_data: [ user_id, positive item_id, 99 negative item_ids ]

    test_data[:, :2] = lines[test_index][:, :2]
    # test_index: where the user_id first appears in lines
    # test_data[:,:2]: [user id, item id]

    for i in range(max_user_id):
        item_neg_map = user_item_neg_map[i]
        item_neg = random.sample(item_neg_map, test_negs)
        # item_neg: test_negs(default 99) item_ids that are neg per user
        test_data[i, 2:] = item_neg


    ########################################
    ### get infer data
    infer_length = max_user_id * max_item_id - len(lines)
    # infer_length = 22869871 (number of items that are not rated)

    infer_data = np.zeros((infer_length, 2), dtype = np.int32)
    count = 0
    for user_id in user_item_neg_map:
        neg_item_list = list(user_item_neg_map[user_id])
        for neg_item in neg_item_list:
            infer_data[count] = [user_id, neg_item]
            count += 1


    ########################################
    ### normalize
    train_data[:, 2] = train_data[:, 2] * 1.0 / max_rating

    ### numpy to torch
    train_data, test_data, infer_data = \
        torch.LongTensor(train_data), torch.LongTensor(test_data), torch.LongTensor(infer_data)

    ### get loader
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
