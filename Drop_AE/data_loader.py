import torch
import torch.utils.data as data
import numpy as np
import random


class Dataset(data.Dataset):
    def __init__(self, lines, num_items):
        self.lines = lines
        self.num_items = num_items

    def __getitem__(self, index):
        line = self.lines[index]
        line = line.split(",")[1:]
        movie_vector = np.zeros(self.num_items)
        for i in line:
            movie_vector[int(i)] = 1.0
        movie_vector = torch.Tensor(movie_vector)
        return movie_vector

    def __len__(self):
        return len(self.lines)


def get_loader(data_path, batch_size = 10000, num_workers = 2):
    with open(data_path, 'r') as f:
        lines = f.readlines()

    train_lines = lines[:300000]
    valid_lines = lines[300000:393186]
    test_lines = lines[393186:]

    train_data = Dataset(train_lines,8926)
    valid_data = Dataset(valid_lines,8926)
    test_data = Dataset(test_lines,8926)

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
