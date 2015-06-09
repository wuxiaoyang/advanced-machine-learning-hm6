import numpy as np
import cPickle as pickle

import SGD_lasso
import matplotlib.pyplot as plt

N, P = 50, 10

def data_gen():

    X = np.random.rand( N, P )
    w = np.random.rand( P, 1 )
    w[0 : P/2] = 0

    for i in range(10):
        np.random.shuffle(w)

    T = np.dot(X,w)

    n2 = np.linalg.norm( T , ord=2)
    m = np.random.normal(0 , 0.01* (n2**2) /N , size = (N,1))

    y = T+m
    return y , X , w

def main():

    y, X , w = data_gen()

    print w-SGD_lasso.solve(X,y)

if __name__ == '__main__':
    main()
