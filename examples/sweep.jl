using PosteriorBounds
using GaussianProcesses
using Random
using LinearAlgebra
using Distributions

# Initialize the GP
Random.seed!(35)
# Training data
n = 100;                          #number of training points
x = 2π * rand(2, n);              #predictors
obs_noise = 0.01
y = sin.(x[1, :] .* x[2, :]) + obs_noise * randn(n);   #regressors
logObsNoise = log10(obs_noise)

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0, 0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

gp = GP(x, y, mZero, kern, logObsNoise)       #Fit the GP

# Setup minimal GP
gp_ex = PosteriorBounds.PosteriorGP(
    gp.dim,
    gp.nobs,
    gp.x,
    gp.cK,
    Matrix{Float64}(undef, gp.nobs, gp.nobs),
    UpperTriangular(zeros(gp.nobs, gp.nobs)),
    inv(gp.cK),
    gp.alpha,
    PosteriorBounds.SEKernel(gp.kernel.σ2, gp.kernel.ℓ2)
)

# Test point-valued bounds
x_t = gp.x[:, 1]
x_L = [0.2, 0.2]
x_U = [0.25, 0.25]
x_test = [0.25; 0.25]

theta_vec, theta_vec_train_squared = PosteriorBounds.theta_vectors(x, gp_ex.kernel)

# Testing out the point-wise bounds
A, B, C, D = PosteriorBounds.calculate_μ_bound_values(gp_ex.alpha, theta_vec, theta_vec_train_squared, x_L, x_U, gp_ex.x)
res1 = PosteriorBounds.μ_bound_point(x_test, theta_vec, A, B, C, D)

# Testing out the point-wise bounds
Ã, B̃, C̃, D̃ = PosteriorBounds.calculate_μ_bound_values(gp_ex.alpha, theta_vec, theta_vec_train_squared, x_L, x_U, gp_ex.x, upper_bound_flag=true)
res2 = PosteriorBounds.μ_bound_point(x_test, theta_vec, Ã, B̃, C̃, D̃, upper_bound_flag=true)

# Testing out the point-wise bounds
Aσ, Bσ, Cσ, Dσ = PosteriorBounds.calculate_σ2_bound_values(gp_ex.K_inv, theta_vec, theta_vec_train_squared, x_L, x_U, gp_ex.x)
res_σ1 = PosteriorBounds.σ2_bound_point(x_test, theta_vec, Aσ, Bσ, Cσ, Dσ)


N_samps = 100
μ_bound_norms = Vector{Float64}(undef, N_samps) 
σ2_vals = Vector{Float64}(undef, N_samps) 

for i=1:N_samps
    xs = [rand(Uniform(x_L[1], x_U[1])); rand(Uniform(x_L[2], x_U[2])) ]
    μ_lb = PosteriorBounds.μ_bound_point(xs, theta_vec, A, B, C, D) 
    μ_ub = PosteriorBounds.μ_bound_point(xs, theta_vec, Ã, B̃, C̃, D̃, upper_bound_flag=true)
    σ2 = PosteriorBounds.σ2_bound_point(xs, theta_vec, Aσ, Bσ, Cσ, Dσ) 
    μ_bound_norms[i] = norm(μ_lb - μ_ub) 
    σ2_vals[i] = σ2
end

using UnicodePlots
plt = histogram(μ_bound_norms, title="μ Bounds Norms, |LB - UB|")
show(plt)
plt2 = histogram(σ2_vals, title="σ2 Upper Bounds")
show(plt2)
