using PosteriorBounds 
using Test

@testset "LearningAbstractions.jl" begin
    # Write your tests here.
    include("gp_bounding_tests.jl")
    include("gp_bounding_tests3d.jl")
end
