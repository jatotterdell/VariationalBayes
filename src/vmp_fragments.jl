function G_VMP(v, Q, r, s)
    d = size(Q)[2]
    v1 = v[1:d]
    v2 = v[d+1:d+d^2]
    V = inv(reshape(v2, d, d))
    return -1/8*tr(Q*V*(v1*v1'*V - 2.0*I) .- 0.5*r'*V*v1 .- 0.5*s)
end


"""
    GaussianPriorFragment(μ, Σ)

```jldoctest
julia> GaussianPriorFragment(zeros(2), zeros(2,2)+I)
6-element Array{Float64,1}:
  0.0
  0.0
 -0.5
 -0.0
 -0.0
 -0.5
```
"""
function GaussianPriorFragment(μ, Σ)
    Σ⁻¹ = inv(Σ)
    η = [Σ⁻¹*μ; -0.5vec(Σ⁻¹)]
    return η
end

"""
    GaussianLikelihoodFragment(n, XtX, Xty, yty, η̃1, η̃2)

Returns the natural parameter contribution from a Gaussian likelihood.
"""
function GaussianLikelihoodFragment(n, XtX, Xty, yty, η̃1, η̃2)
    η1 = (η̃2[1] + 1) / η̃2[2] * [Xty; -0.5*vec(XtX)]
    η2 = [-n/2; G_VMP(η̃1, XtX, Xty, yty)]
    return [η1, η2]
end

"""
    GaussianCommonParameters(η)

Convert natural Gaussian parameters to common parameters.
"""
function GaussianCommonParameters(η)
    n = length(η)
    d = Integer((sqrt(4n + 1) - 1) / 2);
    tmp = inv(vec⁻¹(η[d+1:d+d^2]))
    μ = -0.5*tmp*η[1:d]
    Σ = -0.5*tmp
    return [μ, Σ]
end


"""
    GaussianEntropy(θ::Array)

θ = [μ, Σ] are the common parameters of the Gaussian density.
"""
function GaussianEntropy(θ::Array)
    d = length(θ[1])
    return 0.5*(d*(1 + log(2π)) + logdet(θ[2]))
end
"""
    GaussianEntropy(θ::Array)

η are the natural parameters of the Gaussian density.
"""
function GaussianEntropy(η::Vector)
    θ = GaussianCommonParameters(η)
    return GaussianEntropy(θ)
end
