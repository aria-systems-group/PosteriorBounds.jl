from juliacall import Main as jl
import numpy as np
import time
jl.seval("using PosteriorBounds")
compute_μ_bounds = jl.seval("PosteriorBounds.compute_μ_bounds_bnb_tmp")
compute_σ_bounds = jl.seval("PosteriorBounds.compute_σ_bounds_bnb_tmp")
theta_vectors = jl.seval("PosteriorBounds.theta_vectors")
scale_cKinv = jl.seval("PosteriorBounds.scale_cK_inv")

# GPyTorch Stand-Ins
x = np.array([[1.0,  1.5,  2.0,  0.5], [1.0,  1.5,  2.0,  0.5]]) 

K_inv = np.array([[9.96315,  -8.36825,   3.31192,  -4.9394],
                  [-8.36825,   9.96315,  -4.9394,    3.31192], 
                  [3.31192,  -4.9394,    3.68006,  -1.12941], 
                  [-4.9394,    3.31192,  -1.12941,   3.68006]])

K = np.array([[1.0183156388887342, 0.7788007830714049, 0.36787944117144233, 0.7788007830714049], 
              [0.7788007830714049, 1.0183156388887342, 0.7788007830714049, 0.36787944117144233],
              [0.36787944117144233, 0.7788007830714049, 1.0183156388887342, 0.10539922456186433],
              [0.7788007830714049, 0.36787944117144233, 0.10539922456186433, 1.0183156388887342]])

alpha = np.array([-1.8559087932502434,5.267947387565121, -4.120835360168291, 0.18574839074863536])

x_L = np.array([0.3, 0.3])
x_U = np.array([0.5, 0.5])

sig2 = 1.0
l2 = 1.0
theta_vec, theta_vec_train_squared = theta_vectors(x, l2) 

# Test mean bound
compute_μ_bounds(x, K_inv, alpha, sig2, l2, x_L, x_U, theta_vec_train_squared, theta_vec, bound_epsilon=1e-3)
start = time.process_time()
res = compute_μ_bounds(x, K_inv, alpha, sig2, l2, x_L, x_U, theta_vec_train_squared, theta_vec, bound_epsilon=1e-3)
print(time.process_time() - start)
print(res)

# Test sigma bound
sig2_noise = 0.01
K_inv_scaled = scale_cKinv(K, sig2, sig2_noise)

compute_σ_bounds(x, K, K_inv, alpha, sig2, l2, x_L, x_U, theta_vec_train_squared, theta_vec, K_inv_scaled, bound_epsilon=1e-3)
start = time.process_time()
res = compute_σ_bounds(x, K, K_inv, alpha, sig2, l2, x_L, x_U, theta_vec_train_squared, theta_vec, K_inv_scaled, bound_epsilon=1e-3)
print(time.process_time() - start)
print(res)
