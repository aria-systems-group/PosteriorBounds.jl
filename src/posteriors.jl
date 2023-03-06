abstract type Kernel end

struct PosteriorGP
    dim::Int
    nobs::Int
    x::Matrix{Float64}
    cK::Matrix{Float64}
    cKchol::Matrix{Float64}
    cKcholut::UpperTriangular{Float64, Matrix{Float64}}
    K_inv::Matrix{Float64}
    alpha::Vector{Float64}
    kernel::Kernel
end

"""
Calculate the value of the posterior mean function asusming zero prior mean.
"""
function compute_μ!(μ_h, K_h, x_train::Matrix{Float64}, kernel::Kernel, α::Vector{Float64}, x_pred)
    PosteriorBounds.cov!(K_h, kernel, x_train, x_pred)
    mul!(μ_h, K_h', α)
    return μ_h  
end

function compute_μ!(μ_h, K_h, gp::PosteriorGP, x_pred)
    compute_μ!(μ_h, K_h, gp.x, gp.kernel, gp.alpha, x_pred)
    return μ_h
end

"""
Compute cholesky factors for computing σ
"""
function compute_factors!(gp::PosteriorGP)
    gp.cKchol[:] = gp.cK
    cholesky!(gp.cKchol)
    gp.cKcholut.data[:] = gp.cKchol
end

"""
Compute a single value of σ using GP components
"""
function compute_σ2!(σ2_h, gp::PosteriorGP, x_pred)
    Kcross = Array{Float64}(undef, (gp.nobs, 1))
    cov!(Kcross, gp.kernel, gp.x, x_pred) 
    cov!(σ2_h, gp.kernel, x_pred, x_pred)
    LinearAlgebra.ldiv!(transpose(gp.cKcholut), Kcross)
    σ2_h .-= Kcross'Kcross
    return σ2_h[1]
end

