struct SEKernel <: Kernel
    σ2::Float64
    ℓ2::Float64
end

"""
Squared-exponential kernel explicit function.
"""
function kernel_fcn(x, y, kernel::SEKernel)
    nr = 0
    for i in eachindex(x)
       nr += (x[i] - y[i])^2 
    end
    return kernel.σ2*exp(-nr/(2. * kernel.ℓ2))
end

"""
Calculate the theta vectors given a kernel and dataset.
"""
function theta_vectors(x, kernel::SEKernel)
    return theta_vectors(x, kernel.ℓ2)
end

"""
Calculate the theta vectors given a kernel and dataset.
"""
function theta_vectors(x, ℓ2::Float64)
    dim = size(x,1)
    nobs = size(x,2)
    theta = ones(dim) * 1 ./ (2ℓ2)
    theta_sq = zeros(nobs);
    for i = 1:nobs
        @views theta_sq[i] = transpose(theta) * (x[:, i].^2)
    end   
    return theta, theta_sq
end

"Computes the lower bound of the posterior mean function of a Gaussian process in an interval."
function compute_μ_lower_bound(gp, x_L, x_U, theta_vec_train_squared, theta_vec, 
                               b_i_vec::Vector{Float64}, dx_L::Vector{Float64}, dx_U::Vector{Float64}, H::Vector{Float64}, f::Matrix{Float64}, x_star_h::Vector{Float64}, quad_vec::Vector{Float64}, bi_x_h::Matrix{Float64}, α_temp::Vector{Float64},
                               K_h::Matrix{Float64}, mu_post::Matrix{Float64}; upper_flag=false)
    # Set minmax_factor to -1 if maximizing
    minmax_factor = upper_flag ? -1. : 1.
    x_train = gp.x # Confirmed
    n = size(x_train,1) # Dimension of input
    α_temp .= gp.alpha .* gp.kernel.σ2 
    α_temp *= minmax_factor
    
    H, f, C, a_i_sum = calculate_components(α_temp, theta_vec_train_squared, theta_vec, x_train, x_L, x_U, n, b_i_vec, dx_L, dx_U, H, f, bi_x_h)
    f_val = separate_quadratic_program(H, f, x_L, x_U, x_star_h, quad_vec)
    x_mu_lb = hcat(x_star_h) # TODO: get around hcat?
    
    lb = minmax_factor*(f_val + C + a_i_sum)
    compute_μ!(mu_post, K_h, gp, x_mu_lb)
    ub = mu_post[1]
    
    if upper_flag
        return x_mu_lb, ub, lb
    else
        return x_mu_lb, lb, ub
    end
end

"Computes the upper bound of the posterior covariance function of a Gaussian process in an interval."
function compute_σ_upper_bound(gp, x_L, x_U, cK_inv_scaled, theta_vec_train_squared, theta_vec, 
    b_i_vec::Vector{Float64}, dx_L::Vector{Float64}, dx_U::Vector{Float64}, H::Vector{Float64}, f::Matrix{Float64}, x_star_h::Vector{Float64}, z_i_vector::Matrix{Float64}, quad_vec::Vector{Float64}, bi_x_h::Matrix{Float64}, sigma_post::Matrix{Float64}
    ; min_flag=false)
    minmax_factor = min_flag ? -1. : 1.
    x_train = gp.x # Confirmed
    m = gp.nobs # Confirmed
    n = gp.dim # Dimension of input
    
    sigma_prior = gp.kernel.σ2 # confirmed

    @views for idx=1:m
        z_i_vector[idx, :] .= compute_z_intervals(x_train[:, idx], x_L, x_U, theta_vec, n, dx_L, dx_U) 
    end
    
    a_i_sum = 0. 
    b_i_vec[:] .= 0

    # For each training point
    for idx=1:(m::Int)
        for subidx=1:(idx::Int)
            z_il_L = z_i_vector[idx, 1] + z_i_vector[subidx, 1]
            z_il_U = z_i_vector[idx, 2] + z_i_vector[subidx, 2] 
            a_i, b_i = linear_lower_bound(minmax_factor * cK_inv_scaled[idx, subidx], z_il_L, z_il_U) 
            b_i_vec[idx] += b_i 
            if subidx < idx
                a_i_sum += a_i
                b_i_vec[subidx] += b_i
            end
            a_i_sum += a_i 
        end
    end

    # Hessian object, with respect to each "flex" point
    H .= 4*sum(b_i_vec)*theta_vec   # nx1 vector
    mul!(bi_x_h, b_i_vec', x_train')
    @tullio f[i] = -4*theta_vec[i] .* bi_x_h[i]  
    C = 0.    
    for idx=1:m
       C += 2 * b_i_vec[idx] * theta_vec_train_squared[idx] 
    end
    f_val = separate_quadratic_program(H, f, x_L, x_U, x_star_h, quad_vec)
    x_σ_ub = hcat(x_star_h) 
    σ2_ub = sigma_prior*(1.0 - minmax_factor*(f_val + C + a_i_sum))

    if σ2_ub < 0 && !min_flag
        @warn "Something went wrong bounding σ, use the trivial σ upper bound!"
    end 

    compute_σ2!(sigma_post, gp, x_σ_ub)

    if min_flag
        return x_σ_ub, σ2_ub, sigma_post[1]
    else
        return x_σ_ub, sigma_post[1], σ2_ub
    end
end

function calculate_components(α_train::Vector{Float64}, theta_vec_train_squared, theta_vec, x_train::Matrix{Float64}, x_L, x_U, n::Int, 
                              b_i_vec::Vector{Float64}, dx_L::Vector{Float64}, dx_U::Vector{Float64}, H::Vector{Float64}, f::Matrix{Float64}, bi_x_h::Matrix{Float64})
    a_i_sum = 0. 
    b_i_vec_sum = 0.
    C = 0.
    
    for idx=1:length(α_train)  
        @views z_i_L, z_i_U = compute_z_intervals(x_train[:, idx], x_L, x_U, theta_vec, n, dx_L, dx_U)           
        a_i, b_i = linear_lower_bound(α_train[idx], z_i_L, z_i_U ) # Confirmed!     
        b_i_vec[idx] = b_i
        b_i_vec_sum += b_i
        a_i_sum += a_i 
        C += b_i * theta_vec_train_squared[idx]
    end

    # Hessian object, with respect to each "flex" point
    H .= 2*b_i_vec_sum.*theta_vec   # nx1 vector
    mul!(bi_x_h, b_i_vec', x_train')
    @tullio f[i] = -2*theta_vec[i] .* bi_x_h[i]  

    return H, f, C, a_i_sum
end

"""
Calculate values and vector for bounding μ over an interval.
"""
function calculate_μ_bound_values(α_vec::AbstractArray, θ_vec::AbstractArray, θx2_vec::AbstractArray, x_L::AbstractArray, x_U::AbstractArray, x_train::AbstractArray; upper_bound_flag=false)
    # TODO: add preallocation support
    minmax_factor = upper_bound_flag ? -1 : 1
    a_sum = 0
    nobs = length(α_vec)
    b_vec = zeros(nobs)
    b_vec_sum = 0
    C = 0
    
    dim = length(θ_vec)
    dx_L = zeros(dim)
    dx_U = zeros(dim)

    for idx=1:nobs 
        @views z_i_L, z_i_U = compute_z_intervals(x_train[:, idx], x_L, x_U, θ_vec, dim, dx_L, dx_U)           
        a_i, b_i = linear_lower_bound(minmax_factor * α_vec[idx], z_i_L, z_i_U) # Confirmed!     
        b_vec[idx] = b_i
        b_vec_sum += b_i
        a_sum += a_i 
        C += b_i * θx2_vec[idx]
    end

    bx_vec = b_vec'*x_train'
    D = 2* θ_vec .* bx_vec

    return minmax_factor*a_sum, minmax_factor*b_vec_sum, minmax_factor*C, minmax_factor*D
end

"""
Calculate bound on μ at a single point given necessary values and vectors 
"""
function μ_bound_point(x::AbstractArray, θ_vec::AbstractArray, A::Float64, B::Float64, C::Float64, D::AbstractArray)
    return A + C + B*(x'*diagm(θ_vec)*x)[1] - (D*x)[1]
end

"""
Calculate values and vector for bounding σ over an interval.
"""
function calculate_σ2_bound_values(cK_inv_scaled::AbstractArray, θ_vec::AbstractArray, θx2_vec::AbstractArray, x_L::AbstractArray, x_U::AbstractArray, x_train::AbstractArray; min_flag=false)
    a_sum = 0
    nobs = size(x_train, 2)
    b_vec = zeros(nobs)
    C = 0

    minmax_factor = min_flag ? -1. : 1.
    
    dim = length(θ_vec)
    dx_L = zeros(dim)
    dx_U = zeros(dim)

    z_vec = zeros(nobs, 2)
    @views for idx=1:nobs
        z_vec[idx, :] .= compute_z_intervals(x_train[:, idx], x_L, x_U, θ_vec, dim, dx_L, dx_U) 
    end

    # For each training point
    for idx=1:(nobs::Int)
        for subidx=1:(idx::Int)
            z_il_L = z_vec[idx, 1] + z_vec[subidx, 1]
            z_il_U = z_vec[idx, 2] + z_vec[subidx, 2] 
            a_i, b_i = linear_lower_bound(minmax_factor * cK_inv_scaled[idx, subidx], z_il_L, z_il_U) 
            a_sum += a_i 
            b_vec[idx] += b_i 
            if subidx < idx
                a_sum += a_i
                b_vec[subidx] += b_i
            end
        end
    end
    bx_vec = b_vec'*x_train'
    D = 4* θ_vec .* bx_vec

    C = 0.    
    for idx=1:nobs
       C += 2 * b_vec[idx] * θx2_vec[idx] 
    end

    return minmax_factor*a_sum, minmax_factor*2*sum(b_vec), minmax_factor*C, minmax_factor*D 
end

"""
Calculate bound on σ at a single point given necessary values and vectors 
"""
function σ2_bound_point(x::AbstractArray, θ_vec::AbstractArray, A::Float64, B::Float64, C::Float64, D::AbstractArray; σ_prior=1.0)
    return σ_prior - (A + C + B*(x'*diagm(θ_vec)*x)[1] - (D*x)[1])
end

function compute_z_intervals(x_i, x_L, x_U, theta_vec, n::Int, dx_L::Vector{Float64}, dx_U::Vector{Float64})
    z_i_L = 0.
    dx_L .= (x_i .- x_L).^2       # TODO: This still takes much time, improve further
    dx_U .= (x_i .- x_U).^2
    z_i_U = 0.
    @inbounds for idx=1:n
        if x_L[idx] > x_i[idx] || x_i[idx] > x_U[idx]
            minval = dx_L[idx] < dx_U[idx] ? dx_L[idx] : dx_U[idx] 
            z_i_L += theta_vec[idx] * minval
        end
        z_i_U += theta_vec[idx]*max(dx_L[idx], dx_U[idx])
    end

    return z_i_L, z_i_U
end

function linear_lower_bound(α::Float64, z_i_L::Float64, z_i_U::Float64)
 # Now compute the linear under approximation (inlined for computational reasons)
    if α >= 0.
        z_i_avg = (z_i_L + z_i_U)/2
        e_avg = exp(-z_i_avg)
        αe = e_avg*α
        a_i = (1 + z_i_avg)*αe
        b_i = -αe
    else
        dz = z_i_L - z_i_U 
        ezL = exp(-z_i_L) 
        ezU = exp(-z_i_U) 
        de = ezL - ezU
        b_i = α*(de)/(dz)
        a_i = α*ezL - z_i_L*b_i
    end
    
    return a_i, b_i
end

"A simple quadratic program solver."
function separate_quadratic_program(H::Vector{Float64}, f::Matrix{Float64}, x_L, x_U, x_star_h::Vector{Float64}, quad_vec::Vector{Float64}; C=0.)

    # By default, set the optimal points to the lower bounds
    x_star_h .= x_L
    f_val = 0.    # Value at x*
    n = length(x_L) # Number of dimensions. 
    calc_f_part(ddf::Float64, df::Float64, point::Float64) = 0.5*ddf*point.^2 + df*point
    
    for idx=1:n
        x_critic = -f[idx]/H[idx]
        if H[idx] >= 0 && (x_critic <= x_U[idx]) && (x_critic >= x_L[idx])
            x_star_h[idx] = x_critic
            f_val_partial = calc_f_part(H[idx], f[idx], x_critic)
        else
            quad_vec[1] = calc_f_part(H[idx], f[idx], x_L[idx])
            quad_vec[2] = calc_f_part(H[idx], f[idx], x_U[idx])   
            f_val_partial = minimum(quad_vec)
            if f_val_partial == quad_vec[2]
                x_star_h[idx] = x_U[idx]
            end
        end
        f_val += f_val_partial
    end
    
    return f_val + C
end

"""
Calculate the explicit covariance matrix.
"""
function cov!(K, kernel::SEKernel, x1, x2)
    for i in axes(K, 1)
        for j in axes(K, 2)
            @views K[i,j] = kernel_fcn(x1[:,i], x2[:,j], kernel)
        end
    end
    return K
end