abstract type Kernel end

struct PosteriorGP
    dim::Int
    nobs::Int
    x::Matrix{Float64}
    K_inv::Matrix{Float64}
    alpha::Vector{Float64}
    kernel::Kernel
end

"""
Calculate the value of the posterior mean function asusming zero prior mean.
"""
function predict_μ!(μ_h, K_h, x_train::Matrix{Float64}, kernel::Kernel, α::Vector{Float64}, x_pred)
    PosteriorBounds.cov!(K_h, kernel, x_train, x_pred)
    mul!(μ_h, K_h', α)
    return μ_h  
end

function predict_μ!(μ_h, K_h, gp::PosteriorGP, x_pred)
    predict_μ!(μ_h, K_h, gp.x, gp.kernel, gp.alpha, x_pred)
    return μ_h
end


