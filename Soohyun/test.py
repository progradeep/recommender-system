import numpy as np
import torch
import h5py

npy = np.array([1,2])
np.save("num.npy", npy)
a, b = np.load("num.npy")
print(a, b)