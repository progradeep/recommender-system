import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim # 30

        self.embedding_user = nn.Embedding(num_embeddings = self.num_users, embedding_dim = self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings = self.num_items, embedding_dim = 10)

        self.embedding_genre = nn.Linear(21, 9)
        self.embedding_dir = nn.Linear(3026, 10)

        self.affine_output = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.logistic = nn.Sigmoid()
        self.mseloss = nn.MSELoss()

    def forward(self, user_indices, item_indices, b, g, d):
        user_embedding = self.embedding_user(user_indices)

        item_pref = self.embedding_item(item_indices)
        g_embedding = self.embedding_genre(g.float())
        d_embedding = self.embedding_dir(d.float())
        b = b.view(-1, 1)
        print(item_pref.shape, g_embedding.shape, d_embedding.shape, b.shape)

        item_embedding = torch.cat((item_pref,b.float(),g_embedding,d_embedding),dim=1)

        print("U",user_embedding.shape)
        print("I",item_embedding.shape)

        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits).squeeze()

        return rating


class MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(MLP, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim

        self.embedding_user = nn.Embedding(num_embeddings = self.num_users, embedding_dim = self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings = self.num_items, embedding_dim = self.latent_dim)

        self.fc1 = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.affine_output = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.logistic = nn.Sigmoid()
        self.mseloss = nn.MSELoss()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim = -1)  # the concat latent vector
        vector = self.fc1(vector)
        vector = nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits).squeeze()

        return rating


"""
class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        config = self.config
        config['latent_dim'] = config['latent_dim_mlp']
        mlp_model = MLP(config)
        if config['use_cuda'] is True:
            mlp_model.cuda()
        resume_checkpoint(mlp_model, model_dir=config['pretrain_mlp'], device_id=config['device_id'])

        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        config['latent_dim'] = config['latent_dim_mf']
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat([mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)
"""
