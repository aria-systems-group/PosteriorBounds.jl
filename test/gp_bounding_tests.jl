using PosteriorBounds 
using GaussianProcesses
using Random
using Test
using LinearAlgebra

@testset "jl" begin

    # Initialize the GP
    Random.seed!(35)
    # Training data
    x = [1.0 1.5 2.0 0.5; 1.0 1.5 2.0 0.5]
    obs_noise = 0.01
    y = sin.(x[1,:].*x[2,:]) #+ obs_noise*randn(n);   #regressors
    logObsNoise = log10(obs_noise)

    #Select mean and covariance function
    mZero = MeanZero()                   #Zero mean function
    kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

    gp = GP(x,y,mZero,kern,logObsNoise)       #Fit the GP

    # Test kernel function
    @test PosteriorBounds.kernel_fcn([0., 0.], [1., 1.], SEKernel(1.0, 1.0)) ≈ 0.3678794411714422

    # Test μ prediction
    x_test = [1.11, 1.11]
    K_h = zeros(gp.nobs,1)
    mu_h = zeros(1,1)

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
    PosteriorBounds.compute_factors!(gp_ex)
    PosteriorBounds.compute_μ!(mu_h, K_h, gp_ex, hcat(x_test))
    σ2_h = Matrix{Float64}(undef, 1, 1) 
    PosteriorBounds.compute_σ2!(σ2_h, gp_ex, hcat(x_test))
    @test mu_h[1] ≈ 0.9528022664167798
    @test σ2_h[1] ≈ 0.014003315803827743

    # Compare to prediction from GaussianProcess.jl
    μgp, σgp = GaussianProcesses.predict_f(gp, hcat(x_test)) 
    @test mu_h[1] ≈ μgp[1]
    @test σ2_h[1] ≈ σgp[1]

    # Preallocated arrays for memory savings 
    m_sub = gp.nobs
    b_i_vec = Array{Float64}(undef, m_sub)
    dx_L = zeros(gp.dim)
    dx_U = zeros(gp.dim)
    H = zeros(gp.dim)
    f = zeros(1, gp.dim)
    x_star_h = zeros(gp.dim)
    vec_h = zeros(gp.dim)
    bi_x_h = zeros(1,gp.dim)
    α_h = zeros(gp.nobs)

    # Test compute_z_intervals
    x_t = gp.x[:,1]
    x_L = [0.3, 0.3]
    x_U = [0.5, 0.5]

    theta_vec, theta_vec_train_squared = PosteriorBounds.theta_vectors(x, gp_ex.kernel)

    # Test split_region
    x_avg = zeros(gp.dim)
    new_regions = PosteriorBounds.split_region!(x_L, x_U, x_avg)
    @test new_regions[1] == [[0.3, 0.3], [0.4, 0.4]]  
    @test new_regions[2] == [[0.4, 0.3], [0.5, 0.4]]  
    @test new_regions[3] == [[0.3, 0.4], [0.4, 0.5]]
    @test new_regions[4] == [[0.4, 0.4], [0.5, 0.5]]

    # Test whole algorithm
    x_best, lbest, ubest = PosteriorBounds.compute_μ_bounds_bnb(gp_ex, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-3, max_flag=false)
    @test x_best[1:2] == [0.3, 0.3]
    @test lbest <= ubest
    @test lbest ≈ 0.0600259356942785
    @test ubest ≈ 0.06058892550429269

    tol = 1e-3
    cK_inv_scaled = PosteriorBounds.scale_cK_inv(gp_ex.cK, 1.0, 10e-3)
    sx_best, slbest, subest = PosteriorBounds.compute_σ_bounds(gp_ex, x_L, x_U, theta_vec_train_squared, theta_vec, cK_inv_scaled; bound_epsilon=tol, max_iterations=100)

    μgp, _ = predict_f(gp, hcat(x_best))
    @test μgp[1] ≈ ubest 
    @test μgp[1] > lbest

    _, σ2gp = predict_f(gp, hcat(sx_best))
    @test subest > σ2gp[1]
    @test abs(subest - σ2gp[1]) < tol
    @test slbest ≈ σ2gp[1]

    x_best, lbest, ubest = PosteriorBounds.compute_μ_bounds_bnb(gp_ex, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-3, max_flag=true)
    @test x_best[1:2] == [0.5, 0.5]
    @test lbest ≈ 0.24400185880540715
    @test ubest ≈ 0.24442227145437612
end
