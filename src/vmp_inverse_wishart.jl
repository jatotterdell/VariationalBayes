"""
    InverseGammaPriorFragment(α, β)

Returns the natural parameters from an Inverse-Gamma prior:
``\\eta = (-\\alpha-1 \\beta)``
```jldoctest
>julia InverseGammaPriorFragment(2.0,5.0)
2-element Array{Float64,1}:
 -3.0
  5.0
```
"""
function InverseGammaPriorFragment(α, β)
    return [-α - 1; β]
end


"""
    InverseWishartPriorFragment(ξ, Λ)

Returns the natural parameters from an Inverse-Wishart prior:
``\\eta = [-(\\xi + d + 1)/2, vec(\\Lambda)/2]``

"""
function InverseWishartPriorFragment(ξ, Λ)
    d = size(Λ)[2]
    η = zeros(d^2 + 1)
    η[1] = -0.5*(ξ + d + 1)
    η[2:d^2+1] = -0.5*vec(Λ)
    return η
end


"""
    IteratedInverseGWishartFragment(G, ξ, η̃₁, η̃₂)

Returns the natural parameters from an IteratedInverseGWishartFragment:
``\\eta = [-(\\xi + d + 1)/2, vec(\\Lambda)/2]``

"""
function IteratedInverseGWishartFragment(G, ξ, η̃₁, η̃₂)
    η₁₁ = η̃₁[1]
    η₂₁ = η̃₂[1]
    d = length(η̃₁) - 1
    ω = isdiag(G) ? 1.0 : (d + 1.0) / 2.0
    Ω₁ = (η₁₁ + ω) * inv(vec⁻¹(η̃₁[2:d+1]))
    Ω₂ = (η₂₁ + ω) * inv(vec⁻¹(η̃₂[2:d+1]))
    η₁ = [-0.5*(ξ + 2) ; -0.5 * vec(Ω₂)]
    η₂ = [-0.5*(ξ + 2 - 2ω) ; -0.5*vec(Ω₁)]
    return [η₁, η₂]
end


"""
    InverseGWishartCommonParameters(η)

Convert from natural parameters to common parameters in InverseGWishart.
"""
function InverseGWishartCommonParameters(η)
    η₁ = η[1]
    η₂ = η[2:length(η)]
    ξ = -2 - 2η₁
    Λ = -2vec⁻¹(η₂)
    return [ξ, Λ]
end
