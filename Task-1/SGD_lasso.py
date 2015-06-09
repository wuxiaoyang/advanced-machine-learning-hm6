import numpy as np
import sklearn as skl

max_iter = 1000

def solve(X,y):

    N,P = X.shape
    assert( y.shape == (N,1) )

    a, b , eps = 0, 0, 0.01

    # Init

    w_t , w_t1 , alpha_t = 1 ,0 ,0 

    # Iteration

    t = 0
    while np.linalg.norm(w_t - w_t1) > eps and t < max_iter:

        D = -2 * X.T * ( y- np.dot(X , w_t.T) ) + r * np.sgn(w_t.T).sum()
        alpha_t = a/(t+b)
        w_t1 , w_t = w_t - alpha_t * D, w_t1
        t += 1

    print w_t1
    return w_t1

def main():
    pass

if __name__ == '__main__':
    main()
