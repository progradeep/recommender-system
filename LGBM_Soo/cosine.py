import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import sklearn.preprocessing as pp

"""
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
"""

# score(i, x) = (itemsSimilarity(x, record(i, 1)) + … + itemsSimilarity(x, record(i, M(i)))) /
# (itemsSimilarity(x, item(1)) + … + itemsSimilarity(x, item(K)))
# (단, 식 표현 편의를 위해 itemsSimilarity(x, x) = 0 으로 가정한 식입니다.)
cos = np.load("../../data/cosine_sim.npy")
test = pd.read_csv("../../data/KISA_TBC_VIEWS_UNIQ_TEST.csv")

def calculate_sim(row):
    global cos
    user_id = row['USER_ID'].values[0]
    items = row['MOVIE_ID'].values

    out = []

    for i in range(cos.shape[0]):
        score = 0.
        for m in items:
            score += cos[i,int(m)]
        score /= sum(cos[i,:])

        out.append(score)

    return user_id+sorted(range(len(out)), key=lambda i: out[i], reverse=True)[:50]


grouped = test.groupby(by='USER_ID').apply(calculate_sim).tolist()

