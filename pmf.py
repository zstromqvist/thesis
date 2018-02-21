"""
Class for probabilistic matrix factorization for sparse matrices

"""
# import ctypes
import numpy as np
import math


class pmf:
    """
    Calculates the pmf of a sparse matrix

    Parameters
	----------

    x = a sparse matrix of ratings
    k = # of features
    beta = learning parameter for ..
    alpha = learning parameter for ...
    min_improv = minimun improvement for the algo
    min_epoch = minimum # of epoch
    max_epoch = maximum # of epochs

    """


    def __init__(self, x, k = 10, beta = 0.01, alpha = 0.01, 
        min_improv = 0.000001, min_epoch = 100, max_epoch = 500):

        self.x = x       
        self.k = k
        self.beta = beta
        self.alpha = alpha
        self.min_improv = min_improv
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch


    # getter and setters
    # to come...

    # basic probabilistic matrix factorization, no baseline or fancy stuff
    def pmf(self):


        U = np.ones((self.x.shape[0], self.k)) * 0.1
        V = np.ones((self.x.shape[1], self.k)) * 0.1
        p = np.dot(U, V.T)

        for f in range(self.k):

            p = np.dot(U,V.T)
            epoch = 0
            l_error = 0
            d_error = 0

            while epoch < self.max_epoch:
                error = np.subtract(self.x, p)
                U_tmp = U

                # update features
                for i in range((self.x.shape[1])):
                    for j in range(self.x.shape[0]):
                        if self.x[j,i] > 0:
                            delta_u = self.alpha * (error[j,i] * V[i,f] - self.beta * U[j,f])
                            U[j,f] = U[j,f] + delta_u

                for k in range(self.x.shape[0]):
                    for l in range(self.x.shape[1]):
                        if self.x[k,l] > 0:
                            delta_v = self.alpha * (error[k,l] * U_tmp[k,f] - self.beta * V[l,f])
                            V[l,f] = V[l,f] + delta_v


                # update errors
                p = np.dot(U,V.T)
                n_error = math.sqrt(np.sum(np.power((x-p), 2))/x.size)
                d_error = abs(l_error - n_error)

                l_error = n_error
                epoch += 1
    
    return(p)



    # utils
    # to come...
