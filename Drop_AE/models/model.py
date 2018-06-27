import torch
import torch.nn as nn


class Drop_AE(nn.Module):
    def __init__(self, num_items, latent_dim, drop_rate):
        super(Drop_AE, self).__init__()
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(num_items, latent_dim)
        self.linear2 = nn.Linear(latent_dim, num_items)

        self.dropout = nn.Dropout(p = drop_rate)

    def forward(self, x):
        x = self.dropout(x)

        x = self.linear1(x)
        x = nn.ReLU(x)

        x = self.linear2(x)
        x = nn.Sigmoid(x)

        return x
