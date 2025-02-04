import numpy as np
import torch 
import matplotlib.pyplot as plt
from matplotlib import cm
import jax.numpy as jnp # this is a thin wrapper to NumPy within JAX
from jax import grad, hessian
import step_length
import branin

bounds_branin = np.array([[-5., 0], [10., 15.]])

func  = branin        # function I am minimizing

x0    = np.array([[6., 14.]]) # starting point
g_inf = 10            # starting gradient infinity norm
eps   = 1e-5          # tolerance for convergence
k     = 0             # iteration counter
maxiters = 20         # maximum number of iterations
xk    = x0            # starting point 

bounds= bounds_branin # optimization variable bounds
fk    = func(xk)

# empty lists to store optimization history
ginf_sd_b = []        # first-order optimality
xk_sd_b   = []        # iterate history
xk_sd_b.append(x0)    ## include the starting point
ncalls_sd_b = []      # number of function calls
f_sd_b      = []      # objective history

np.set_printoptions(precision=3)
print(f'starting point x0: {xk}, f0: {fk}')
print('----------------------------------')

while g_inf >= eps and k < maxiters:    
    gk = np.asarray(grad(func)(xk.squeeze()) )
    pk = -gk/np.linalg.norm(gk)                # steepest-descent direction
    sl, nfcalls, ngcalls = step_length(func, xk, pk)             # calculate step length
    alpha = sl.line_search()        
    xk    = xk + alpha * pk                       # new iterate
    fk    = func(xk)                              # evaluate f at new iterate
    g_inf = np.linalg.norm(gk, ord=np.inf)     # check first-order optimality (gradient)
    
    k += 1
    ncalls_sd_b.append(nfcalls + 1)
    ginf_sd_b.append(g_inf)
    xk_sd_b.append(xk)
    f_sd_b.append(fk)

    print(f'iteration {k}, nfcalls: {nfcalls + 1}, ngcalls: {ngcalls}, alpha: {alpha:1.7f}, xk: {xk.squeeze()}, fk: {fk.item():2.6f}, gradient norm: {g_inf:2.6f}')