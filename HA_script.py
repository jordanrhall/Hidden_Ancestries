### Hidden Ancestries Script

import numpy as np
import scipy as scipy
from scipy.optimize import minimize
import timeit


### data_processer: (D, k, obs) -> (A, taf)


### A data-processing function that takes 3 inputs: 

 ## 1. A user-input genetic data array D -- needs to be processed via pandas on the python side!!! 
 ## Note that D should be of size Nx(k+1+?) where we know there are N=number of SNPs rows and at least k columns, with at least one more for the total allele frequencies, and potentially ? other reference info (like SNP location, chromosome number, etc...)

 ## 2. the number of ancestries, k=2,3,4,..., 

 ## 3. obs = 1, 2, 3, etc is which column to pull the observed/taf from *after* the MAFs
 ## So for example, if the taf is stored in the first column after the MAFs, then obs=1. This is the default for HA (below).

### and returns 2 outputs:

 ## 1. Genetic data in an input array "A" size Nxk containing N SNPs (these are the rows), and k ancestries (these are the columns);

 ## 2. The total allele frequency called "taf" which should be an Nx1 vector

def data_processor(D,k,obs):
    N = np.shape(D)[0] # N=number of SNPs!
    A = np.zeros((N,k))
    taf = np.zeros((N,1))
    names = D.columns # collect list of column names

    # Assume that D has 3 columns we do not need (columns 0, 1 and 2 in python)...
    # Then we can grab out the MAFs
    for i in range(2,2+k):
        A[:,i-2] = D[names[i]]

    taf[:,0] = D[names[k+1+obs]] # + 1 because python starts indexing at 0.

    return A, taf




### HA : (D, k, x_guess, obs=1) -> (x_answer, n_iterations, time)


### A generalized function that takes 4 inputs: 
 
 ## 1. The genetic data frame D (usually a CSV read in through pandas)
 
 ## 2. A starting guess "x_guess" which should be a kx1 vector. Default is 1/k*(1,1,...,1).

 ## 3. The number of ancestries in the input data, k

 ## 4. obs = 1, 2, 3, etc is which column to pull the observed/taf from *after* the MAFs
 ## So for example, if the taf is stored in the first column after the MAFs, then obs=1. This is the default for HA (below).
    
### and returns 3 outputs:

 ## 1. The hidden proportions of every ancestry in the data as a kx1 vector
    
 ## 2. The number of iterations that SLSQP did as a scalar value

 ## 3. The run time of the algoirthm as a scalar value, measured in seconds


def HA(D=None, k=None, x_guess=None, obs=1):

    if D is None:
        print('Please specify Nxk data matrix D.')
        return

    if abs(np.shape(np.shape(D))[0]-2)>0:
        print('Please ensure that data matrix D is size Nxk.')
        return
   
    if k is None:
        print('Please specify k, the number of ancestries.')
        return

    if isinstance(k,int)==False:
        print('Please ensure that k is an integer.')
        return
    elif k <=0:
        print('Please ensure that k is a positive integer.')
        return

    if x_guess is None:
        x_guess=np.transpose(1/k*np.ones((k,1)))

    if abs(np.shape(x_guess)[0]-1)>0 and abs(np.shape(x_guess)[1]-1)>0:
        print('Please ensure that initial iterate x_guess is a vector, size kx1 or 1xk.')
        return

    if abs(np.shape(x_guess)[1]-k)>0:
        x_guess=np.transpose(np.copy(x_guess))

    # Use the data_processor to take the info we need out of the data frame D
    data_array=data_processor(D,k,obs)
    A = data_array[0]
    taf = data_array[1]

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
