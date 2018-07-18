import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import sklearn.preprocessing as pp


dataroot = "../../data/"
def cosine_similarities(sp_mat):
    col_normed_mat = pp.normalize(sp_mat, axis=0)
    sim = col_normed_mat.T * col_normed_mat

    return sim.todense()

data  = pd.read_csv("../../data/KISA_TBC_VIEWS_UNIQ.csv")

user = data['USER_ID'].values
movie = data['MOVIE_ID'].values

target = [1] * len(user)
data = csc_matrix((target,(user, movie)))
print("csc matrix built")

sp.save_npz(dataroot+"csc_mat_for_cos.npz",data)
print("mat saved!")


cos = cosine_similarities(data)
np.save(dataroot+"cosine_sim.npy",cos)
