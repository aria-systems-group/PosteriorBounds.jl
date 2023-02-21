from juliacall import Main as jl
import numpy as np
jl.seval("using PosteriorBounds")
compute_μ_bounds = jl.seval("PosteriorBounds.compute_μ_bounds_bnb")
PosteriorGP = jl.seval("PosteriorBounds.PosteriorGP")
SEKernel = jl.seval("PosteriorBounds.SEKernel")

# GPyTorch Stand-Ins
x = np.array([[1.0,  1.5,  2.0,  0.5], [1.0,  1.5,  2.0,  0.5]]) 

K_inv = np.array([[9.96315,  -8.36825,   3.31192,  -4.9394],
                  [-8.36825,   9.96315,  -4.9394,    3.31192], 
                  [3.31192,  -4.9394,    3.68006,  -1.12941], 
                  [-4.9394,    3.31192,  -1.12941,   3.68006]])

alpha = np.array([[1.8559087932502434],
 [-5.267947387565121],
  [4.120835360168291],
 [-0.18574839074863536]])

gp = PosteriorGP(
    2,
    4,
    x,
    K_inv, 
    alpha,
    SEKernel(1.0, 1.0)
)

x_L = np.array([0.3, 0.3])
x_U = np.array([0.5, 0.5])
theta_vec = np.ones(gp.dim) * 1 / (2*gp.kernel.ℓ2)
theta_vecT = np.transpose(theta_vec)
theta_vec_train_squared = np.zeros(gp.nobs)
for i in range(0, gp.nobs):
    theta_vec_train_squared[i] = np.dot(theta_vecT, np.square(gp.x[:, i]))

res = compute_μ_bounds(gp, x_L, x_U, theta_vec_train_squared, theta_vec)