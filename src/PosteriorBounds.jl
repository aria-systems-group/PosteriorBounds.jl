module PosteriorBounds

# using GaussianProcesses
using LinearAlgebra
using Random
using Distributions
using StaticArrays
using Tullio

include("bnb.jl")
include("posteriors.jl")
include("squared_exponential.jl")

export SEKernel

end # module PosteriorBounds
