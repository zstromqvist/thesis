"""
Class for probabilistic matrix factorization for sparse matrices

"""
# import ctypes
import numpy as np
import math


class Pmf():
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
    P = prediction matrix

    """


    def __init__(self, x = None, k = 10, beta = 0.01, alpha = 0.01, 
        min_improv = 0.000001, min_epoch = 100, max_epoch = 200, P = None):

        self._x = x       
        self._k = k
        self._beta = beta
        self._alpha = alpha
        self._min_improv = min_improv
        self._min_epoch = min_epoch
        self._max_epoch = max_epoch
        self._P = P


    # getter and setters
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x):
        self._x = x

    @property
    def k(self):
        return self._k
    
    @k.setter
    def k(self, k):
        self._k = k

    @property
    def beta(self):
        return self._beta
    
    @beta.setter
    def beta(self, beta):
        self._beta = beta
    
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
    
    @property
    def min_improv(self):
        return self._min_improv
    
    @min_improv.setter
    def min_improv(self, min_improv):
        self._min_improv = min_improv
    
    @property
    def min_epoch(self):
        return self._min_epoch
    
    @min_epoch.setter
    def min_epoch(self, min_epoch):
        self._min_epoch = min_epoch
    
    @property
    def max_epoch(self):
        return self._max_epoch
    
    @max_epoch.setter
    def max_epoch(self, max_epoch):
        self._max_epoch = max_epoch

    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, P):
        self._P = P
    
    # basic probabilistic matrix factorization, no baseline or fancy stuff
    # Gee, this is so brute force it's ridiculous. No premature optimization here
    def pmf(self):

        # initialize the vector of features and compute prediction matrix
        U = np.ones((self._x.shape[0], self._k)) * 0.1
        V = np.ones((self._x.shape[1], self._k)) * 0.1
        p = np.dot(U, V.T)

        # update loop for each feature
        for f in range(self._k):

            p = np.dot(U,V.T)
            epoch = 0
            #l_error = 0
            #d_error = 0

            # this is the meat
            # need to speed this shit up, look into ctypes
            # do a number of epochs for each feature
            while epoch < self._max_epoch:
                error = np.subtract(self._x, p)
                U_tmp = U

                # update features
                for i in range((self._x.shape[1])):
                    for j in range(self._x.shape[0]):
                        if self._x[j,i] > 0:
                            delta_u = self._alpha * (error[j,i] * V[i,f] - self._beta * U[j,f])
                            U[j,f] = U[j,f] + delta_u

                for k in range(self._x.shape[0]):
                    for l in range(self._x.shape[1]):
                        if self._x[k,l] > 0:
                            delta_v = self._alpha * (error[k,l] * U_tmp[k,f] - self._beta * V[l,f])
                            V[l,f] = V[l,f] + delta_v


                # update errors
                p = np.dot(U,V.T)
                #n_error = math.sqrt(np.sum(np.power((x-p), 2))/x.size)
                #d_error = abs(l_error - n_error)

                #l_error = n_error
                epoch += 1 # update epoch

                # print out current feature and epoch
                print("Feature: {}/{}, epoch: {}/{}".format(f+1, self._k, epoch, self._max_epoch) )
        
        p = np.around(p,2) # round to two decimals
        self._P = p
        return self



    # utils
    # to come...
    """
    Function for creating goal matrix for testing

    Function for making goal matrix sparse

    Function to calculate error, both for the sparse matrix and for the goal matrix

    """
