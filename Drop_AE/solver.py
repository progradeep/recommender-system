import math
import os
import numpy as np

import torch
from torch import optim
import torch.nn.functional as F

from torch.autograd import Variable

from models.model import Drop_AE

import pandas as pd


class Solver(object):
    def __init__(self, config):

        self.model_type = config.model_type

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
            self.model = Drop_AE(8259, self.latent_dim, 0.8)

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


    def apk(self, actual, predicted, k=50):
        """
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        #if not actual:
        #    return 0.0

        return score / min(len(actual), k)

    def mapk(self, actual, predicted, k=50):
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        Parameters
        ----------
        actual : list
                 A list of lists of elements that are to be predicted
                 (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The mean average precision at k over the input lists
        """
        return np.mean([self.apk(a, p, k) for a, p in zip(actual, predicted)])

    def calculate_map(self, actual, drop_data, predicted, k=50):
        actual = np.nonzero(actual)
        predicted = predicted * (1-drop_data)
        _, predicted = torch.topk(predicted, k)
        mAP = self.mapk(actual, predicted, k)

        return mAP

    def train(self, train_loader, valid_loader):

        print("Start Train!!")
        print()

        total_step = len(train_loader)

        step = 0
        for epoch in range(self.num_epochs):
            for i, data in enumerate(train_loader):
                step += 1
                self.model.train()

                data = self.to_variable(data)
                drop_data = self.model.dropout(data)
                outputs = self.model(drop_data)

                loss = F.mse_loss(outputs, data.data)

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
                for i, data in enumerate(valid_loader):
                    if i == 1: break
                    data = self.to_variable(data)
                    outputs = self.model(data)
                
                    print('Epoch [%d/%d], MAP: %.4f' % (epoch + 1, self.num_epochs,
                                                        self.calculate_map(data, outputs, self.topk)))

            model_path = os.path.join(self.save_path, 'model-%d.pkl' % (epoch + 1))
            torch.save(self.model, model_path)

    def test(self, test_loader):
        print("Start Test!!")
        print()

        step = 0
        for i, data in enumerate(test_loader):
            step += 1
            self.model.eval()

            data = self.to_variable(data)
            outputs = self.model(data)
            outputs = outputs * (1-data)

            _, outputs = torch.topk(outputs, self.topk)

            outputs = pd.DataFrame(outputs)

            if step == 1:
                outputs.to_csv(self.output_path, index = False)
            else:
                prev_output = pd.read_csv(self.output_path)
                prev_output.append(outputs)
                prev_output.to_csv(self.output_path, index = False)
            
        print("Save file in {}".format(self.output_path))
    # def infer(self, infer_loader):
    #
    #     print("Start Inference!!")
    #     print()
    #     self.model.eval()
    #
    #     user_item_recommender_table = torch.zeros(self.num_users, self.num_items)
    #
    #     for i, data in enumerate(infer_loader):
    #         data = self.to_variable(data)
    #         user = data[:, 0]
    #         item = data[:, 1]
    #
    #         score = self.model(user, item).cpu()
    #
    #         user_item_recommender_table[user.data, item.data] = score.data
    #
    #     _, topk_item_for_user = torch.topk(user_item_recommender_table, k = self.topk)
    #
    #     user_item_recommender_table = user_item_recommender_table.numpy()
    #     topk_item_for_user = topk_item_for_user.numpy()
    #
    #     np.save(self.infer_path + "/user_item_recommender_table.npy", user_item_recommender_table)
    #     np.save(self.infer_path + "/topk_item_for_user.npy", topk_item_for_user)
    #
    #     print("Save file in {}".format(self.infer_path))
    #     print()
