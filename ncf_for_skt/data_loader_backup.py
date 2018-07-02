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
#       -> test_data(num_user, 2 + 99): [usr id, pos item id, neg item ids]
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
    # convert file to matrix and rename ids
    user_id_to_num = {}
    item_id_to_num = {}
    num_to_user_id = []
    num_to_item_id = []

    # delete first line
    lines = lines[1:]

    for i in range(len(lines)):
        lines[i] = lines[i].split(",")           # "::" for movielens , "," for csv file
        lines[i] = lines[i][:2]  # remove timestamp
        lines[i] = [int(value) for value in lines[i]]
        lines[i].append(1)        # dont use for movielens

        if not lines[i][0] in user_id_to_num:  # rename user id
            m = len(num_to_user_id)
            user_id_to_num[lines[i][0]] = m
            num_to_user_id.append(lines[i][0])
            lines[i][0] = m
        else:
            m = user_id_to_num[lines[i][0]]
            lines[i][0] = m

        if not lines[i][1] in item_id_to_num:  # rename item id
            m = len(num_to_item_id)
            item_id_to_num[lines[i][1]] = m
            num_to_item_id.append(lines[i][1])
            lines[i][1] = m
        else:
            m = item_id_to_num[lines[i][1]]
            lines[i][1] = m
    
    num_user = len(num_to_user_id)
    num_item = len(num_to_item_id)
    print("num_user : {}".format(num_user))
    print("num_item : {}".format(num_item))
    max_rating = 1
    print("max_rating : {}".format(max_rating))

    ########################################
    ### find negative map: user_item_neg_map
    # find user item mapping
    user_item_map = {}
    #delete_list = []
    for i, line in enumerate(lines):
        user_id = line[0]
        item_id = line[1]
        if not user_id in user_item_map:
            ## when a certain user_id first appears,
            # make a set containing item_id and append it to map
            user_set = set([item_id])
            user_item_map[user_id] = user_set

        else:
            #if item_id in user_item_map[user_id]:
            #    delete_list.append(i)
            ## if the user_id is already in map,
            # add the item_id to the user's user_set
            user_item_map[user_id].add(item_id)
    # user_item_map: {0:{0,2691,1245,2686},1:{1356,3452,743}, ,,, }
    # delete overlap data
    #new_lines = [line for i, line in enumerate(lines) if not i in delete_list]
    #lines = new_lines
   
    all_item_map = set(list(range(num_item)))
    ########################################
    ### convert numpy array
    # divide data into train and test data
    lines = np.array(lines)
    # lines.shape: (changeable) (1000209, 3)
    test_index = np.zeros(num_user, dtype = np.int32)
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
    ### add negative samples to train data
    shape = train_lines.shape
    # shape.shape: (994169, 3)

    neg_data = np.zeros((shape[0] * train_negs, shape[1]), dtype = np.int32)
    # neg_data.shape: (994169 * 4, 3), all zeros


    print('add random sampled neg data to train set')
    for i, line in enumerate(train_lines):
        user_id = line[0]
        item_neg_map = all_item_map - user_item_map[user_id]

        item_neg = random.sample(item_neg_map, train_negs)
        # item_neg: choose train_negs(default 4) neg items from each user

        for j in range(train_negs):
            neg_data[i * train_negs + j] = [user_id, item_neg[j], 0]
            # Ex) i=0, j=0: nd[0*4 + 0] = [0, random[0], 0]
            # Ex) i=0, j=1: nd[0*4 + 1] = [0, random[1], 0]
            # Ex) i=1, j=0: nd[1*4 + 0] = [0, random[0], 0]
        if user_id % 1000 == 0: print(user_id)
    train_data = np.concatenate((train_lines, neg_data), axis = 0)
    # train_data.shape: (994169 * 5, 3)


    ### add negative samples to test data
    test_data = np.zeros((num_user, test_negs + 2), dtype = np.int32)
    # test_data.shape: (num_user, 2 + 99)
    # test_data: [ user_id, positive item_id, 99 negative item_ids ]

    test_data[:, :2] = lines[test_index][:, :2]
    # test_index: where the user_id first appears in lines
    # test_data[:,:2]: [user id, item id]

    print('add random sampled neg data to test set')
    for user_id in range(num_user):
        item_neg_map = all_item_map - user_item_map[user_id]
        item_neg = random.sample(item_neg_map, test_negs)
        # item_neg: test_negs(default 99) item_ids that are neg per user
        test_data[user_id, 2:] = item_neg
        if user_id % 1000 == 0: print(user_id)

    ########################################
    ### normalize
    train_data[:, 2] = train_data[:, 2] * 1.0 / max_rating

    ### numpy to torch
    train_data, test_data, infer_data = \
        torch.LongTensor(train_data), torch.LongTensor(test_data), torch.LongTensor(infer_data)

    ### get loader
    train_data = User_Item_Dataset(train_data)
    test_data = User_Item_Dataset(test_data)

    train_loader = data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    test_loader = data.DataLoader(test_data, batch_size = num_user, shuffle = True, num_workers = num_workers)

    print()
    print("Complete Data Processing!!")
    print()

    return num_user, num_item, train_loader, test_loader, num_to_user_id, num_to_item_id


def get_infer_loader(data_path, train_negs = 4, test_negs = 99, batch_size = 100, num_workers = 2):
    ########################################
    ### load file
    # each line contains (user_id, item_id, rating, timestamp)
    with open(data_path, 'r') as f:
        lines = f.readlines()
    ### find user_max, item_max, rating_max
    # convert file to matrix and rename ids
    user_id_to_num = {}
    item_id_to_num = {}
    num_to_user_id = []
    num_to_item_id = []

    # delete first line
    lines = lines[1:]

    for i in range(len(lines)):
        lines[i] = lines[i].split(",")           # "::" for movielens , "," for csv file
        lines[i] = lines[i][:2]  # remove timestamp
        lines[i] = [int(value) for value in lines[i]]
        lines[i].append(1)        # dont use for movielens

        if not lines[i][0] in user_id_to_num:  # rename user id
            m = len(num_to_user_id)
            user_id_to_num[lines[i][0]] = m
            num_to_user_id.append(lines[i][0])
            lines[i][0] = m
        else:
            m = user_id_to_num[lines[i][0]]
            lines[i][0] = m

        if not lines[i][1] in item_id_to_num:  # rename item id
            m = len(num_to_item_id)
            item_id_to_num[lines[i][1]] = m
            num_to_item_id.append(lines[i][1])
            lines[i][1] = m
        else:
            m = item_id_to_num[lines[i][1]]
            lines[i][1] = m
    
    num_user = len(num_to_user_id)
    num_item = len(num_to_item_id)
    print("num_user : {}".format(num_user))
    print("num_item : {}".format(num_item))
    max_rating = 1
    print("max_rating : {}".format(max_rating))

    ########################################
    ### find negative map: user_item_neg_map
    # find user item mapping
    user_item_map = {}
    #delete_list = []
    for i, line in enumerate(lines):
        user_id = line[0]
        item_id = line[1]
        if not user_id in user_item_map:
            ## when a certain user_id first appears,
            # make a set containing item_id and append it to map
            user_set = set([item_id])
            user_item_map[user_id] = user_set

        else:
            #if item_id in user_item_map[user_id]:
            #    delete_list.append(i)
            ## if the user_id is already in map,
            # add the item_id to the user's user_set
            user_item_map[user_id].add(item_id)
    # user_item_map: {0:{0,2691,1245,2686},1:{1356,3452,743}, ,,, }
    # delete overlap data
    #new_lines = [line for i, line in enumerate(lines) if not i in delete_list]
    #lines = new_lines
   
    print('count infer_length')
    infer_length = 0
    N = 393186
    # find user item negative mapping
    all_item_map = set(list(range(num_item)))
    for user_id in user_item_map:
        # all_item_map: {0,1,2, ,,, num_item}
        if user_id > N:
            neg_item_list = list(all_item_map - user_item_map[user_id])
            infer_length += len(neg_item_list)
            #for neg_item in neg_item_list:
            #    infer_data.append([user_id, neg_item])
            if user_id % 10000 == 0: print(user_id)	
        # user_item_neg_map: items_id that users didn't answer

    infer_data = np.zeros([infer_length, 2], dtype = np.int32)

    count = 0
    print('make infer_data')
    for user_id in user_item_map:
        # all_item_map: {0,1,2, ,,, num_item}
        if user_id > N:
            neg_item_list = list(all_item_map - user_item_map[user_id])
            for neg_item in neg_item_list:
                infer_data[count, 0] = user_id
                infer_data[count, 1] = neg_item
                count += 1
            if user_id % 10000 == 0: print(user_id)
        # user_item_neg_map: items_id that users didn't answer

    ### numpy to torch
    infer_data = torch.LongTensor(infer_data)

    ### get loader
    infer_data = User_Item_Dataset(infer_data)

    infer_loader = data.DataLoader(infer_data, batch_size = batch_size, shuffle = False, num_workers = num_workers)

    print()
    print("Complete Data Processing!!")
    print()

    return num_user, num_item, infer_loader, num_to_user_id, num_to_item_id




if __name__ == "__main__":
    data_path = "/root/data/KISA_TBC_VIEWS_MERGE.csv"
    num_user, num_item, train_loader, test_loader, infer_loader, num_to_user_id, num_to_item_id \
        = get_loader(data_path=data_path, batch_size = 3)
    print(num_user)
    print()
    print(num_item)
    print()
    for i, data in enumerate(train_loader):
        if i > 0:
            break
        else:
            print(data)
            print()
    for i, data in enumerate(test_loader):
        if i > 0:
            break
        else:
            print(data)
            print()
    for i, data in enumerate(infer_loader):
        if i > 0:
            break
        else:
            print(data)
            print()
