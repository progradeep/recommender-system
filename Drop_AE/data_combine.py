import numpy as np

f = open("../../data/short200000.csv","r")
d1 = f.readlines()
f.close()
g = open("../../data/short_end.csv","r")
d2 = g.readlines()
g.close()

print([1]+[2])
combined = d1 + d2
print(len(d1), len(d2), len(combined))

with open("../../data/ae_total.csv","w") as ff:
    for line in combined:
	    ff.write(line)

