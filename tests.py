import numpy as np

x = np.random.random_integers(1,5, size=(500,300))

y = np.random.uniform(0,1,size=(500,300))

# sparsity
z = y < 0.05

user_rating_mat = np.multiply(x,z)

print(user_rating_mat[:10,:10])
print(np.sum(z.astype(int)))
print(500*300)
print(np.sum(z.astype(int))/(300*500))




