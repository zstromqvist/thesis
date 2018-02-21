import numpy as np
from pmf import Pmf

rows = 15
cols = 15
x = np.random.random_integers(1,5, size=(rows,cols))
y = np.random.uniform(0,1,size=(rows,cols))

# sparsity
z = y < 0.10

user_rating_mat = np.multiply(x,z)

t = Pmf()
t.x = user_rating_mat
t.k = 10
t.pmf()
print(t.P)
print(t.x)



