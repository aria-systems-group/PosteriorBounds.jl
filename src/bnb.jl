"""
Compute an bounds on the posterior mean value in an interval. Defaults to bounding the max value.
"""
function compute_μ_bounds_bnb_tmp(x, K_inv, alpha, σ2, ℓ2, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-2, max_flag=false, image_prealloc=nothing)

    gp = PosteriorGP(
        size(x, 1),
        size(x, 2),
        convert(Matrix, x),
        convert(Matrix, K_inv),
        convert(Vector, alpha),
        SEKernel(σ2, ℓ2)
    )

    return compute_μ_bounds_bnb(gp, convert(Vector, x_L), convert(Vector, x_U), convert(Vector,theta_vec_train_squared), convert(Vector, theta_vec); max_iterations=max_iterations, bound_epsilon=bound_epsilon, max_flag=max_flag, image_prealloc=image_prealloc)
end

"""
Compute an bounds on the posterior mean value in an interval. Defaults to bounding the max value.
"""
function compute_μ_bounds_bnb(gp, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-2, max_flag=false, image_prealloc=nothing)
    # If no preallocation object is provided, preallocate
    # This could be done more elegantly, but leave it for now...
    if isnothing(image_prealloc)      
        dx_L = zeros(gp.dim)
        dx_U = zeros(gp.dim)
        H = zeros(gp.dim)
        f = zeros(1, gp.dim)
        x_star_h = zeros(gp.dim)
        vec_h = zeros(2)
        bi_x_h = zeros(1,gp.dim)
        b_i_vec = Array{Float64}(undef, gp.nobs)
        α_h = zeros(gp.nobs)
        K_h = zeros(gp.nobs,1)
        mu_h = zeros(1,1)
    else
        dx_L = image_prealloc.dx_L 
        dx_U = image_prealloc.dx_U 
        H = image_prealloc.H 
        f = image_prealloc.f 
        x_star_h = image_prealloc.x_star_h 
        vec_h = image_prealloc.vec_h 
        bi_x_h = image_prealloc.bi_x_h 
        b_i_vec = image_prealloc.b_i_vec 
        α_h = image_prealloc.α_h 
        K_h = image_prealloc.K_h 
        mu_h = image_prealloc.mu_h 
    end
    
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
