module VariationalBayes

export vmp_lm

include("vmp_inverse_wishart.jl")
include("vmp_lm.jl")
include("vmp_fragments.jl")

end # module
