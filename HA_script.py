### Hidden Ancestries Script (v 3.0)


import numpy as np
import scipy as scipy
from scipy.optimize import minimize
import timeit
import pandas as pd


### data_processor: (file, file_format, k, obs) -> (A, taf)

### A data-processing function that takes 4 inputs: 

 ## 1. file: A user-input genetic data "file" -- will be processed via pandas below!!! 
 ## Must be a .txt or a .csv file (with the .txt or .csv as the last four characters of the actual file name). See data formatting standards for more info about required rows/columns.

 ## 2. file_format: The "file_format" of file, as a string. Default is 'tab' which is short for tab-delimited text files. Can also choose 'csv' for CSV files.

 ## 3. k: the number of reference ancestries, k=2,3,4,...

 ## 4. obs: The value obs = 1, 2, 3, etc is which column to pull the observed/taf from *after* the reference ancestries
 ## So for example, if the observed is stored in the first column after the reference ancestries, then obs=1. This is the default for HA (below).

### and returns 2 outputs:

 ## 1. A: Genetic data in an input array "A" size Nxk containing N SNPs (these are the rows), and k reference ancestries (these are the columns);

 ## 2. taf: The observed or "total allele frequency" called "taf" in this code, which should be an Nx1 vector

def data_processor(file, file_format, k, obs):

    # Reads data file in using pandas
    if (file_format=='csv') == True:
        D = pd.read_csv(file)
    else:
        D = pd.read_csv(file, sep='\t')

    # Extract key variables
    N = np.shape(D)[0] # N=number of SNPs!
    A = np.zeros((N,k))
    taf = np.zeros((N,1))
    names = D.columns # collect list of column names

    # Here we make sure that naming conventions are correct in given data and matches the number of reference ancestries, k
    for i in range(0,k):
        if ('ref' in names[i+5]) == False:
            print('Please ensure that all K reference populations are labeled with ref. See data formatting standards for more details.')
            return

    # Assume that D has 5 columns we do not need for calculations (columns 0, 1, 2, 3 and 4 in python)...
    # Then we can grab out the reference ancestries
    for i in range(5,5+k):
        A[:,i-5] = D[names[i]]

    taf[:,0] = D[names[k+4+obs]] # + 1 because python starts indexing at 0.

    return A, taf


### HA : (k, x_guess, obs, file, file format) -> (x_answer, n_iterations, time)

### A generalized function that takes 5 inputs: 

 ## 1. k: The number of reference ancestries in the input data

 ## 2. x_guess: A starting guess, which should be a kx1 vector. Default is 1/k*(1,1,...,1).

 ## 3. obs: The value obs = 1, 2, 3, etc is which column to pull the observed/taf from *after* the reference ancestries
 ## So for example, if the taf is stored in the first column after the reference ancestries, then obs=1. This is the default for HA (below).
 
 ## 4. file: The genetic data frame "file" (usually a tab-delimited text file which we read in through pandas using data processor above)

 ## 5. The "file_format" of file. Should be .csv or .txt as described above. Default assumption is .txt, tab delimited text.
    
### and returns 3 outputs:

 ## 1. x_answer: The hidden proportions of every reference ancestry in the data as a kx1 vector
    
 ## 2. n_iteration: The number of iterations that SLSQP did as a scalar value

 ## 3. time: The run time of the algoirthm as a scalar value, measured in seconds

def HA(k=None, x_guess=None, obs=1, file=None, file_format='tab'):

    # Start the clock!
    start = timeit.default_timer()

    if file is None:
        print('Please specify a genetic data frame.')
        return

    # Use the data_processor to take the info we need out of the data frame D
    data_array = data_processor(file, file_format, k, obs)
    A = data_array[0]
    taf = data_array[1]

    if abs(np.shape(np.shape(A))[0]-2)>0:
        print('Please ensure that data matrix D is size Nxk.')
        return
   
    if k is None:
        print('Please specify k, the number of reference ancestries.')
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

    if abs(np.shape(x_guess)[1]-k)>0:
        print('Please ensure that initial iterate x_guess is a vector, size kx1 or 1xk.')

    if isinstance(obs,int)==False:
        print('Please ensure that obs is an integer.')
        return

    elif obs <=0:
        print('Please ensure that obs is a positive integer.')
        return

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
        d = 0

	# We still need the value of the "inside" of the objective function, so we repeat part of what we did above:        
        for i in range(0,k):
            d = d + x[i]*A[:,i:(i+1)]
        d = d - taf
	# Now d is Nx1 and contains the value of the mixture model minus the total allele frequencies at each SNP

	# Now we form the k entries of the gradient and return that vector        
        for i in range(0,k):
            gradvec[i,:] = np.sum(2*A[:,i:(i+1)]*d, axis=0)
        return gradvec

    # These are wrappers that make our constraints (all proportions must add to 1) and our bounds (all proportions are 0 or greater)
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x,axis=0) -1},)

    bnds = ((0, None),)

    for i in range(0,k-1):
        bnds = bnds + ((0, None),)

    # We now form an answer object which will store all of the outputs to running SLSQP given our inputs above
    ans_obj = scipy.optimize.minimize(obj_fun, x_guess, method='SLSQP', jac=grad_obj_fun, bounds=bnds, constraints=cons, tol=1e-5)
    
    # Stop the clock!
    stop = timeit.default_timer()

    # Difference stop-start tells us run time
    time= stop-start

    # Print results for the user!
    print('Numerical solution via SLSQP, pi_final = ',ans_obj.x)
    print()
    print('Number of SLSQP iterations:',ans_obj.nit)
    print()
    print('Runtime:',time, 'seconds')

    # Return the 3 outputs we wanted, namely: the solution vector, number of iterations, and run time
    return ans_obj.x, ans_obj.nit, time
