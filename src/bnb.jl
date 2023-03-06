"""
Compute an bounds on the posterior mean value in an interval. Defaults to bounding the max value.
"""
function compute_μ_bounds_bnb_tmp(x, K_inv, alpha, σ2, ℓ2, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-2, max_flag=false, prealloc=nothing)

    gp = PosteriorGP(
        size(x, 1),
        size(x, 2),
        convert(Matrix, x),
        Matrix{Float64}(undef, gp.nobs, gp.nobs),
        Matrix{Float64}(undef, gp.nobs, gp.nobs),
        UpperTriangular(zeros(gp.nobs, gp.nobs)),
        convert(Matrix, K_inv),
        convert(Vector, alpha),
        SEKernel(σ2, ℓ2)
    )

    return compute_μ_bounds_bnb(gp, convert(Vector, x_L), convert(Vector, x_U), convert(Vector,theta_vec_train_squared), convert(Vector, theta_vec); max_iterations=max_iterations, bound_epsilon=bound_epsilon, max_flag=max_flag, prealloc=prealloc)
end

"""
Compute an bounds on the posterior variance value in an interval.
"""
function compute_σ_bounds_bnb_tmp(x, K, K_inv, alpha, σ2, ℓ2, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-2, max_flag=false, prealloc=nothing)

    gp = PosteriorGP(
        size(x, 1),
        size(x, 2),
        convert(Matrix, x),
        K,
        Matrix{Float64}(undef, gp.nobs, gp.nobs),
        UpperTriangular(zeros(gp.nobs, gp.nobs)),
        convert(Matrix, K_inv),
        convert(Vector, alpha),
        SEKernel(σ2, ℓ2)
    )
    compute_factors!(gp)

    return compute_σ_bounds_bnb(gp, convert(Vector, x_L), convert(Vector, x_U), convert(Vector,theta_vec_train_squared), convert(Vector, theta_vec); max_iterations=max_iterations, bound_epsilon=bound_epsilon, prealloc=prealloc)
end

struct Preallocs
    dx_L::Vector{Float64} 
    dx_U::Vector{Float64} 
    H::Vector{Float64} 
    f::Matrix{Float64} 
    x_star::Vector{Float64} 
    quad_vec::Vector{Float64} 
    z_i_vector::Matrix{Float64}
    b_i_vec::Vector{Float64} 
    bi_x_h::Matrix{Float64} 
    α_temp::Vector{Float64} 
    K_h::Matrix{Float64} 
    mu_post::Matrix{Float64} 
    sigma_post::Matrix{Float64} 
end

"""
Create preallocation structures for memory efficiency
"""
function preallocate_matrices(dim::Int, nobs::Int)
    prealloc = Preallocs(
        zeros(Float64, dim),        # dx_L
        zeros(Float64, dim),        # dx_U
        zeros(Float64, dim),        # H
        zeros(Float64, 1, dim),     # f
        zeros(Float64, dim),        # x_star
        zeros(Float64, 2),          # quad_vec
        zeros(Float64, nobs, 2),    # z_i_vec
        zeros(Float64, nobs),       # b_i_vec
        zeros(Float64, 1, dim),     # bi_x_h
        zeros(Float64, nobs),       # α_temo 
        zeros(Float64, nobs,1),     # K_h
        zeros(Float64, 1,1),        # mu_post
        zeros(Float64, 1,1)         # sigma_post
    )
    return prealloc
end

"""
Compute an bounds on the posterior mean value in an interval. Defaults to bounding the max value.
"""
function compute_μ_bounds_bnb(gp, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-2, max_flag=false, prealloc=nothing)

    # If no preallocation object is provided, preallocate
    image_prealloc = isnothing(prealloc) ? preallocate_matrices(gp.dim, gp.nobs) : prealloc
        
    dx_L = image_prealloc.dx_L 
    dx_U = image_prealloc.dx_U 
    H = image_prealloc.H 
    f = image_prealloc.f 
    x_star_h = image_prealloc.x_star 
    vec_h = image_prealloc.quad_vec 
    bi_x_h = image_prealloc.bi_x_h 
    b_i_vec = image_prealloc.b_i_vec 
    α_h = image_prealloc.α_temp 
    K_h = image_prealloc.K_h 
    mu_h = image_prealloc.mu_post 

    x_best, lbest, ubest = compute_μ_lower_bound(gp, x_L, x_U, theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, vec_h, bi_x_h, α_h, K_h, mu_h, upper_flag=max_flag)
    if max_flag
        temp = lbest
        lbest = -ubest
        ubest = -temp
    end
    
    candidates = [(x_L, x_U)]
    iterations = 0

    split_regions = nothing
    x_avg = zeros(length(x_L))

    while !isempty(candidates) && iterations < max_iterations
        new_candidates = []
        for extent in candidates
            
            if isnothing(split_regions)
                split_regions = split_region!(extent[1], extent[2], x_avg) 
            else
                split_regions = split_region!(extent[1], extent[2], x_avg, new_regions=split_regions)
            end  

            for pair in split_regions
                x_lb1, lb1, ub1 = compute_μ_lower_bound(gp, pair[1], pair[2], theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, vec_h, bi_x_h, α_h, K_h, mu_h, upper_flag=max_flag)
                if max_flag
                    temp = lb1
                    lb1 = -ub1
                    ub1 = -temp
                end
                
                if ub1 <= ubest
                    ubest = ub1
                    lbest = lb1
                    x_best = x_lb1
                    push!(new_candidates, pair)
                elseif lb1 < ubest   
                    push!(new_candidates, pair)
                end
                
            end
            
        end

        if norm(ubest - lbest) < bound_epsilon
            break
        end
        candidates = new_candidates
        iterations += 1
    end
    if max_flag
        temp = lbest
        lbest = -ubest
        ubest = -temp
    end

    return x_best, lbest, ubest 
end

function compute_σ_ub_bounds(gp, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=10, bound_epsilon=1e-4, prealloc=nothing)
    
    # If no preallocation object is provided, preallocate
    image_prealloc = isnothing(prealloc) ? preallocate_matrices(gp.dim, gp.nobs) : prealloc
        
    dx_L = image_prealloc.dx_L 
    dx_U = image_prealloc.dx_U 
    H = image_prealloc.H 
    f = image_prealloc.f 
    x_star_h = image_prealloc.x_star 
    vec_h = image_prealloc.quad_vec 
    bi_x_h = image_prealloc.bi_x_h 
    b_i_vec = image_prealloc.b_i_vec 
    z_i_vector = image_prealloc.z_i_vector
    sigma_post = image_prealloc.sigma_post

    x_best, lbest, ubest = compute_σ_upper_bound(gp, x_L, x_U, gp.K_inv, theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, z_i_vector, vec_h, bi_x_h, sigma_post)

    candidates = [(x_L, x_U)]
    iterations = 0

    split_regions = nothing
    x_avg = zeros(gp.dim)
   
    while !isempty(candidates) && iterations < max_iterations
        new_candidates = []
        for extent in candidates
            if isnothing(split_regions)
                split_regions = split_region!(extent[1], extent[2], x_avg) 
            else
                split_regions = split_region!(extent[1], extent[2], x_avg, new_regions=split_regions)
            end  
            
            for pair in split_regions
                x_ub1, lb1, ub1 = compute_σ_upper_bound(gp, pair[1], pair[2], gp.K_inv, theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, z_i_vector, vec_h, bi_x_h, sigma_post)
                if lb1 >= lbest
                    lbest = lb1
                    ubest = ub1
                    x_best = x_ub1
                    push!(new_candidates,pair)
                elseif ub1 > lbest
                    push!(new_candidates, pair)
                end

                if norm(ubest - lbest) < bound_epsilon
                    return x_best, lbest, ubest
                end
            end
        end
        
        candidates = new_candidates
        iterations += 1
    end

    return x_best, lbest, ubest
end

"""
Subdivide the input region into smaller regions.
"""
function split_region!(x_L, x_U, x_avg; new_regions=nothing)
    n = length(x_L)
    x_avg .= (x_L .+ x_U)/2

    lowers = [[x_L[i], x_avg[i]] for i=1:n]
    uppers = [[x_avg[i], x_U[i]] for i=1:n]

    if isnothing(new_regions)
        new_regions = [[[lower...], [upper...]] for (lower, upper) in zip(Base.product(lowers...), Base.product(uppers...))] 
    else
        new_regions .= [[[lower...], [upper...]] for (lower, upper) in zip(Base.product(lowers...), Base.product(uppers...))]  
    end

    return new_regions
end
