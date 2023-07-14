module VariationalBayes

using LinearAlgebra, Distributions

export vmp_lm

include("utils.jl")
# include("canonical_parameters.jl")
include("vmp_inverse_wishart.jl")
include("vmp_lm.jl")
include("vmp_fragments.jl")

end # module
