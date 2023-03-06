using PosteriorBounds
using GaussianProcesses
using Random
using Test
using LinearAlgebra

@testset "jl" begin

    # Initialize the GP
    Random.seed!(35)
    # Training data
    n=100;                          #number of training points
    x = 2π * rand(3,n);              #predictors
    obs_noise = 0.01
    y = sin.(x[1,:].*x[2,:]) + obs_noise*randn(n);   #regressors
    logObsNoise = log10(obs_noise)

    #Select mean and covariance function
    mZero = MeanZero()                   #Zero mean function
    kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

    gp = GP(x,y,mZero,kern,logObsNoise)       #Fit the GP

    # Preallocated arrays for memory savings 
    m_sub = gp.nobs
    b_i_vec = Array{Float64}(undef, m_sub)
    dx_L = zeros(gp.dim)
    dx_U = zeros(gp.dim)
    H = zeros(gp.dim)
    f = zeros(1, gp.dim)
    x_star_h = zeros(gp.dim)
    vec_h = zeros(2)
    bi_x_h = zeros(1,gp.dim)
    α_h = zeros(gp.nobs)
    K_h = zeros(gp.nobs,1)
    mu_h = zeros(1,1)

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

    # Test compute_z_intervals
    x_t = gp.x[:,1]
    x_L = [0.3, 0.3, 0.3]
    x_U = [0.5, 0.5, 0.5]

    theta_vec, theta_vec_train_squared = PosteriorBounds.theta_vectors(x, gp_ex.kernel)
    z_interval = @views PosteriorBounds.compute_z_intervals(x_t, x_L, x_U, theta_vec, gp.dim, dx_L, dx_U)
    @test z_interval[1] ≈ 16.7870941340354
    @test z_interval[2] ≈ 18.384475224839704

    α_train = gp.alpha 
    sigma_prior = gp.kernel.σ2 # confirmed
    α_train *= sigma_prior # confirmed

    # Test linear_lower_bound
    a_i, b_i = PosteriorBounds.linear_lower_bound(α_train[1], z_interval[1], z_interval[2]) 
    @test a_i ≈ 3.787943421924841e-7
    @test b_i ≈ -2.0380863586112917e-8

    # Test the whole components
    H, f, C, a_i_sum = PosteriorBounds.calculate_components(α_train, theta_vec_train_squared, theta_vec, gp.x, x_L, x_U, gp.dim, b_i_vec, dx_L, dx_U, H, f, bi_x_h)
    @test H ≈ [-0.09259538702612813, -0.09259538702612813, -0.09259538702612813]
    @test f ≈ [0.20220199653277468 0.09796210866257525 0.18855887451566397]
    @test C ≈ -0.45285779513474134
    @test a_i_sum ≈ 0.35755185118698907

    # Test separate_quadratic_program
    f_val = PosteriorBounds.separate_quadratic_program(H, f, x_L, x_U, x_star_h, vec_h)
    @test x_star_h == [0.3, 0.3, 0.3]
    @test f_val ≈ 0.13411651666477686

    # Test μ prediction
    μ = PosteriorBounds.compute_μ!(mu_h, K_h, gp_ex, hcat(x_star_h))
    @test μ[1] ≈ 0.06789440098151765
end