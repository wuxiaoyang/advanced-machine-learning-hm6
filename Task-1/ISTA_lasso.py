import numpy as np
import itertools
import sklearn as skl

max_iter = 10**7
eps=1e-5

def solve(X,y):

    N, P = X.shape
    L = 1000.0
    lbd = 1.0/N
    
    assert( y.shape == (N,1) )

    w = np.random.rand( P,1 )
    S = np.random.rand( P,1 )

    z = np.zeros( shape = (P,1) )

 #   for t in xrange(max_iter):
    for i in itertools.count():
        
        D = -2 * np.dot(X.T , y - np.dot(X,w) )/N

        S = np.multiply( np.sign(w - D/L) , np.maximum( np.abs(w - D/L) - lbd/L , z  ) )

        if np.linalg.norm(w - S) < eps:
            print 'ISTA converged.'
            break
        w = S

    return w

def main():
    pass

if __name__ == '__main__':
    main()
