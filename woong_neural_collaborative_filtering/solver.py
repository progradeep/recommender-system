import math
import os
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

from models.model import MLP, GMF


class Solver(object):
    def __init__(self, config, num_users, num_items):

        self.model_type = config.model_type

        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = config.latent_dim
        self.lr = config.lr
        self.reg = config.reg
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.num_workers = config.num_workers
        self.save_path = config.save_path
        self.infer_path = config.infer_path
        self.load_path = config.load_path
        self.data_path = config.data_path
        self.log_step = config.log_step
        self.test_step = config.test_step
        self.topk = config.topk
        self.use_gpu = config.use_gpu

        self.build_model()

    def build_model(self):
        if self.load_path == None:
            if self.model_type == 'gmf':
                self.model = GMF(self.num_users, self.num_items, self.latent_dim)
            elif self.model_type == 'mlp':
                self.model = MLP(self.num_users, self.num_items, self.latent_dim)

        else:
            self.model = torch.load(self.load_path)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    self.lr, [self.beta1, self.beta2])
        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()

    def to_variable(self, x):
        if torch.cuda.is_available() and self.use_gpu:
            x = x.cuda()
        return Variable(x)

    def hit_ratio_ndcg_map(self, data):
        user = data[:, 0]
        pos = data[:, 1]
        neg = data[:, 2:]
        batch_size = data.size()[0]
        test_negs = data.size()[1] - 2

        score = torch.zeros(batch_size, test_negs + 1)
        score[:, 0] = self.model(user, pos).data
        for i in range(test_negs):
            score[:, i + 1] = self.model(user, neg[:, i]).data
        _, index = torch.topk(score, k = self.topk)
        # index: indices of largest k elements in score.
        # index.shape: (max_user_id, self.topk)

        zero = torch.zeros(index.size())
        # zero.shape: (max_user_id, self.topk)

        test_in_top_k = (index.float() == zero.float())
        # test_in_top_k: 1 at index == 0, 0 at others
        # test_in_top_k.shape: (max_user_id, self.topk)

        rank = torch.nonzero(test_in_top_k)[:, -1].float()
        # rank: indices of nonzero elements in test_in_top_k
        #       Ex) second row, fifth colmun -> [[ 1, 4] , ,,, ]
        # rank.shape: (changeable) (3508,)

        map = (1 / (rank + 1)).sum() / batch_size

        rank = math.log(2) / torch.log(2 + rank)
        # IDCG: 1/log2(2+0)
        # DCG: 1/log2(2+4)
        # NDCG: log(2)/log(2+4)

        hit_ratio = len(rank) / batch_size
        # hit ratio: number of nonzero elements out of a batch
        ndcg = rank.sum() / batch_size

        return hit_ratio, ndcg, map

    def train(self, train_loader, test_loader):

        print("Start Train!!")
        print()
        total_step = len(train_loader)
        for epoch in range(self.num_epochs):
            for i, data in enumerate(train_loader):

                self.model.train()

                data = self.to_variable(data)

                user_id = data[:, 0]
                item_id = data[:, 1]
                ratings = data[:, 2].float()

                outputs = self.model(user_id, item_id)
                loss = self.model.mseloss(outputs, ratings)

                param_sum = 0
                for param in self.model.parameters():
                    param_sum += (param * param).sum()

                loss += self.reg * param_sum

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % self.log_step == 0:
                    print('Epoch [%d/%d], Step[%d/%d], MSE_loss: %.4f'
                          % (epoch + 1, self.num_epochs, i + 1, total_step, loss))

            if (epoch + 1) % self.test_step == 0:
                self.model.eval()
                for i, data in enumerate(test_loader):
                    data = self.to_variable(data)
                    hit_ratio, ndcg, map = self.hit_ratio_ndcg_map(data)
                    print()
                    print('Epoch [%d/%d], Hit_Ratio: %.4f, NDCG: %.4f, MAP: %.4f'
                          % (epoch + 1, self.num_epochs, hit_ratio, ndcg, map))
                    print()

            model_path = os.path.join(self.save_path, 'model-%d.pkl' % (epoch + 1))
            torch.save(self.model, model_path)

    def infer(self, infer_loader, num_to_user_id, num_to_item_id):

        print("Start Inference!!")
        print()
        self.model.eval()

        user_item_recommender_table = torch.zeros(self.num_users, self.num_items)

        for i, data in enumerate(infer_loader):
            data = self.to_variable(data)
            user = data[:, 0]
            item = data[:, 1]

            score = self.model(user, item).cpu()

            user_item_recommender_table[user.data, item.data] = score.data

        _, topk_item_for_user = torch.topk(user_item_recommender_table, k=self.topk)

        topk_item_for_user_to_id = np.zeros((self.num_users, self.topk + 1))
        # (user_id, item_id1, item_id2, ... item_id10..)
        for user_num, topk_item_num in enumerate(topk_item_for_user):
            for i in range(self.topk + 1):
                if i == 0:
                    topk_item_for_user_to_id[user_num][i] = num_to_user_id[user_num]
                else:
                    topk_item_for_user_to_id[user_num][i] = num_to_item_id[topk_item_num[i - 1]]

        np.save(self.infer_path + "/topk_item_for_user_to_id.npy", topk_item_for_user_to_id)

        print("Save file in {}".format(self.infer_path))
        print()