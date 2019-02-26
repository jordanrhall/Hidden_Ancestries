### Hidden Ancestries Script

### HA : (A, taf, x_guess) -> (x_answer, n_iterations, time)

### A generalized function that takes 3 inputs: 
 
 ## 1. Genetic data in a matrix "A" size Nxk containing N SNPs (these are the rows), and k ancestries (these are the columns);

 ## 2. The total allele frequency called "taf" which should be an Nx1 vector
 
 ## 3. A starting guess "x_guess" which should be a kx1 vector
    
### and returns 3 outputs:

 ## 1. The hidden proportions of every ancestry in the data as a kx1 vector
    
 ## 2. The number of iterations that SLSQP did as a scalar value

 ## 3. The run time of the algoirthm as a scalar value, measured in seconds

import numpy as np
import scipy as scipy
from scipy.optimize import minimize
import timeit

def HA(A, taf, x_guess):
    
    # Grab the number of ancestries
    k=np.shape(A)[1]
    
    # Start the clock!
    start = timeit.default_timer()
    
    # This is the objective function!
    def obj_fun(x):
        b=0
        for i in range(0,k):
            b=b + x[i]*A[:,i:(i+1)]
        b=b-taf
        return np.sum(b**2, axis=0)[0]
    
    # This is the gradient of the objective function!
    def grad_obj_fun(x):

        gradvec = np.zeros((k,1))

        d=0

        for i in range(0,k):
            d=d + x[i]*A[:,i:(i+1)]
        d=d-taf

        for i in range(0,k):
            gradvec[i,:] = np.sum(2*A[:,i:(i+1)]*d, axis=0)
        return gradvec

    # These are wrappers that make our constraints and our bounds

    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x,axis=0) -1},)

    for i in range(0,k-1):
        cons = cons + ({'type': 'ineq', 'fun': lambda x: x[i]},)

    bnds = ((0, None),)

    for i in range(0,k-1):
        bnds = bnds + ((0, None),)

    ans_obj = scipy.optimize.minimize(obj_fun, x_guess, method='SLSQP', jac=grad_obj_fun, bounds=bnds, constraints=cons, tol=1e-5)
    
    stop = timeit.default_timer()
    
    time= stop-start
    
    return ans_obj.x, ans_obj.nit, time