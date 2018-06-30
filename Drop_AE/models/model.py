import torch
import torch.nn as nn


class Drop_AE(nn.Module):
    def __init__(self, num_items, latent_dim, drop_rate):
        super(Drop_AE, self).__init__()
        self.num_items = num_items
        self.latent_dim = latent_dim # 128
        #self.linear1 = nn.Linear(num_items, latent_dim)
        #self.linear2 = nn.Linear(latent_dim, num_items)
        
        self.encoder = nn.Sequential(
		    nn.Linear(num_items, latent_dim * 2), # e1
			nn.SELU(),
			nn.Linear(latent_dim * 2, latent_dim), # e2
            nn.SELU(),
            nn.Linear(latent_dim, latent_dim), # coding layer
			nn.SELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SELU(),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SELU(),
            nn.Linear(latent_dim * 2, num_items),
            nn.Sigmoid(),
        )


        self.dropout = nn.Dropout(p = drop_rate) # dropout rate: 0.8

    def forward(self, x):
        x = self.encoder(x)
        #x = self.linear1(x)
        #x = nn.ReLU(x)
        x = self.decoder(x)

        #x = self.linear2(x)
        #x = nn.Sigmoid(x)

        return x
