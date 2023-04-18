module PosteriorBounds

# using GaussianProcesses
using LinearAlgebra
using LinearAlgebra: BlasReal
using Random
using Distributions
using StaticArrays
using Tullio

include("bnb.jl")
include("posteriors.jl")
include("squared_exponential.jl")

export PosteriorGP
export SEKernel

end # module PosteriorBounds
