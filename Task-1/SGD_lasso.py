import numpy as np
import itertools
import sklearn as skl

max_iter = 10**7

def solve(X,y):

    N, P = X.shape
    assert( y.shape == (N,1) )

    a, b = 1.0, 10.0
    eps = 10.0/max_iter
    #eps = 1e-6
    lbd = 0.2 

    # Init

    w_t , w_t1 = np.random.rand(P,1) , np.random.rand(P,1) 

    # Iteration

    #for t in itertools.count():
    for t in xrange(max_iter):

        D = -2 * np.dot( X.T , ( y - np.dot(X , w_t) ) ) + lbd * np.sign(w_t)
        alpha_t = a/(t+b)

        w_t1 , w_t  = w_t - alpha_t * D, w_t1

        difsign = (w_t1 * w_t>=0)
        w_t1 *= difsign

        if np.linalg.norm(w_t - w_t1) < eps:
            print 'SGD Converged, %d iters.'%t
            break

    return w_t1

def main():
    pass

if __name__ == '__main__':
    main()
