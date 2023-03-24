"""
Compute an bounds on the posterior mean value in an interval. Defaults to bounding the max value.
"""
function compute_μ_bounds_bnb_tmp(x, K_inv, alpha, σ2, ℓ2, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-2, max_flag=false, prealloc=nothing)

    dim = size(x, 1)
    nobs = size(x, 2)
    gp = PosteriorGP(
        dim,
        nobs,
        convert(Matrix, x),
        Matrix{Float64}(undef, nobs, nobs), # uneeded for μ, empty placeholders
        Matrix{Float64}(undef, nobs, nobs),
        UpperTriangular(zeros(nobs, nobs)),
        convert(Matrix, K_inv),
        convert(Vector, alpha),
        SEKernel(σ2, ℓ2)
    )

    return compute_μ_bounds_bnb(gp, convert(Vector, x_L), convert(Vector, x_U), convert(Vector,theta_vec_train_squared), convert(Vector, theta_vec); max_iterations=max_iterations, bound_epsilon=bound_epsilon, max_flag=max_flag, prealloc=prealloc)
end

"""
Compute an bounds on the posterior variance value in an interval.
"""
function compute_σ_bounds_bnb_tmp(x, K, K_inv, alpha, σ2, ℓ2, x_L, x_U, theta_vec_train_squared, theta_vec, cK_inv_scaled; max_iterations=100, bound_epsilon=1e-2, min_flag=false, prealloc=nothing)

    dim = size(x, 1)
    nobs = size(x, 2)
    gp = PosteriorGP(
        dim,
        nobs,
        convert(Matrix, x),
        convert(Matrix, K),
        Matrix{Float64}(undef, nobs, nobs),
        UpperTriangular(zeros(nobs, nobs)),
        convert(Matrix, K_inv),
        convert(Vector, alpha),
        SEKernel(σ2, ℓ2)
    )
    compute_factors!(gp)

    return compute_σ_bounds(gp, convert(Vector, x_L), convert(Vector, x_U), convert(Vector,theta_vec_train_squared), convert(Vector, theta_vec), cK_inv_scaled; max_iterations=max_iterations, bound_epsilon=bound_epsilon, prealloc=prealloc, min_flag=min_flag)
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

function compute_σ_bounds(gp, x_L, x_U, theta_vec_train_squared, theta_vec, cK_inv_scaled; max_iterations=10, bound_epsilon=1e-4, prealloc=nothing, min_flag=false)
   
    @info "This is a σ bounding debug message!"
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

    x_best = nothing
    lbest = -Inf
    ubest = Inf

    if min_flag
        temp = ubest
        ubest = -lbest
        lbest = -temp
    end

    minmax_factor = min_flag ? -1 : 1
    candidates = [(x_L, x_U, minmax_factor*Inf)]
    iterations = 0

    split_regions = nothing
    x_avg = zeros(gp.dim)
    while !isempty(candidates) && iterations < max_iterations
        new_candidates = []
        for extent in candidates

            # Skip candidates according to current best bound
            # if min_flag 
            #     if extent[3] > lbest
            #         continue
            #     end
            # else
            #     if extent[3] < lbest
            #         continue
            #     end
            # end

            if isnothing(split_regions)
                split_regions = split_region!(extent[1], extent[2], x_avg) 
            else
                split_regions = split_region!(extent[1], extent[2], x_avg, new_regions=split_regions)
            end  
            
            for pair in split_regions
                x_ub1, lb1, ub1 = compute_σ_upper_bound(gp, pair[1], pair[2], cK_inv_scaled, theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, z_i_vector, vec_h, bi_x_h, sigma_post, min_flag=min_flag)

                # temp lb fix 
                f_s = 1. / (1 + 1/(iterations*5+1))
                lb1 *= f_s
                if min_flag
                    temp = ub1
                    ub1 = -lb1
                    lb1 = -temp
                end

                if lb1 >= lbest
                    lbest = lb1
                    ubest = ub1
                    x_best = x_ub1
                    push!(new_candidates, (pair[1], pair[2], lbest))
                elseif ub1 > lbest
                    push!(new_candidates, (pair[1], pair[2], lb1))
                end
            end
        end

        if norm(ubest - lbest) < bound_epsilon
            break
        end
        candidates = new_candidates
        iterations += 1
    end

    if min_flag
        temp = ubest
        ubest = -lbest
        lbest = -temp
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
