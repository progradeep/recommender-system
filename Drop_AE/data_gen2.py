from tqdm import tqdm
import numpy as np
import pandas as pd
from time import time

input_path = "../../data/KISA_TBC_VIEWS_UNIQ.csv"
output_path = "../../data/short_end.csv"

with open(input_path, "r") as file:
    data = file.readlines()
    data = np.asarray(data)
data = np.genfromtxt(input_path, delimiter=",",dtype=int)
data = data[1:,:2]

f = open(output_path, 'w')
start = time()

for user in range(200000,len(data)):
    ind = np.where(data[:,0]==user)
    tmp = data[ind[0],1].tolist()
    f.write(str(tmp)[1:-1]+'\n')
    if user % 100 == 0:
	    end = time()
	    print("user", user, "done!",end-start)

print("res done")
f.close()
