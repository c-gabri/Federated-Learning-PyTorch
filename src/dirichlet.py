import torch
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

m = torch.distributions.dirichlet.Dirichlet(torch.tensor([0.5,0.5]))
vect = []
for i in range(50):
    sample = m.sample()
    print(sample)
    vect.append(float(sample[0]))

# print(vect)
# plt.hist(vect, bins = 10)
# plt.show()
dists = [0,1,2,3,4,5,6,7,8,9]
u = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
v = [1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
v = np.ones(10)

u = [0.5,0.2,0.3]
v = [0.5,0.3,0.2]

# create and array with cardinality 3 (your metric space is 3-dimensional and
# where distance between each pair of adjacent elements is 1
dists = [i for i in range(len(u))]
dist = stats.wasserstein_distance(dists, dists, u, v)
print(dist)