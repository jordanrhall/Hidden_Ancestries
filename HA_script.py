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

	# Start the value of the objective function at 0     
	b=0

	# This adds up each k column of A scaled by the k-th ancestry
        for i in range(0,k):
            b=b + x[i]*A[:,i:(i+1)]
	# After the for loop, b is an Nx1 vector which contains the value of the mixture model for all N SNP's

	# Now we subtract off the total allele frequency at each SNP      
	b=b-taf

	# Finally we square every entry of the Nx1 vector b, and add them all up.
	# This is the value of the objective function, which we now return
        return np.sum(b**2, axis=0)[0]
    
    # This is the gradient of the objective function!
    def grad_obj_fun(x):

	# Initiate empty kx1 vector
        gradvec = np.zeros((k,1))

	# Start the value of the gradient entries with 0        
	d=0

	# We still need the value of the "inside" of the objective function, so we repeat part of what we did above:        
	for i in range(0,k):
            d=d + x[i]*A[:,i:(i+1)]
        d=d-taf
	# Now d is Nx1 and contains the value of the mixture model minus the total allele frequencies at each SNP

	# Now we form the k entries of the gradient and return that vector        
	for i in range(0,k):
            gradvec[i,:] = np.sum(2*A[:,i:(i+1)]*d, axis=0)
        return gradvec

    # These are wrappers that make our constraints (all proportions must add to 1) and our bounds (all proportions are 0 or greater)
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x,axis=0) -1},)

    for i in range(0,k-1):
        cons = cons + ({'type': 'ineq', 'fun': lambda x: x[i]},)

    bnds = ((0, None),)

    for i in range(0,k-1):
        bnds = bnds + ((0, None),)

    # We now form an answer object which will store all of the outputs to running SLSQP given our inputs above
    ans_obj = scipy.optimize.minimize(obj_fun, x_guess, method='SLSQP', jac=grad_obj_fun, bounds=bnds, constraints=cons, tol=1e-5)
    
    # Stop the clock!
    stop = timeit.default_timer()

    # Difference stop-start tells us run time
    time= stop-start
    
    # Return the 3 outputs we wanted, namely: the solution vector, number of iterations, and run time
    return ans_obj.x, ans_obj.nit, time
