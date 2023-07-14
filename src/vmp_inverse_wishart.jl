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
    η[1] = -0.5 * (ξ + d + 1)
    η[2:d^2+1] = -0.5 * vec(Λ)
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
    η₁ = [-0.5 * (ξ + 2); -0.5 * vec(Ω₂)]
    η₂ = [-0.5 * (ξ + 2 - 2ω); -0.5 * vec(Ω₁)]
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


"""
    TwoLevelIteratedInverseChiSquared

Algorithm 40
"""
function TwoLevelIteratedInverseChiSquared(y, X, Z, s², ν, 𝔼_inv_a, 𝔼β, 𝕍β, 𝔼γ, 𝕍γ, 𝔼βγ)
    m = length(y) # Number of units
    n = sum(length.(y)) # Number of observations
    κ_σ² = ν + n
    κ_a = ν + 1
    λ_σ² = 𝔼_inv_a
    for i in 1:m
        tmp = y[i] - X[i] * 𝔼β - Z[i] * 𝔼γ
        λ_σ² += tmp' * tmp
        λ_σ² += tr(X' * X * 𝕍β) + tr(Z' * Z * 𝕍γ)
        λ_σ² += 2 * tr(Z' * X * 𝔼βγ[i])
    end
    𝔼_inv_σ² = κ_σ² * inv(λ_σ²)
    λ_a = 𝔼_inv_σ² + inv(ν * s²)
    𝔼_inv_a = κ_a * inv(λ_a)
    return 𝔼_inv_σ², 𝔼_inv_a
end


"""
    TwoLevelIteratedInverseGWishart(s_Σ, ν_Σ, 𝔼_inv_A, μ_γ, Σ_γ)

Algorithm 41
s_Σ > 0 are hyperparameter standard deviations
ν_Σ > 0 is hyperparameter degrees of freedom
𝔼_inv_A is current value for 𝔼(A⁻¹)
μ_γ is current value for 𝔼(γ)
Σ_γ is current value for 𝕍(γ)
"""
function TwoLevelIteratedInverseGWishart(s²_Σ, ν_Σ, 𝔼_inv_A, μ_γ, Σ_γ)
    κ_Σ = ν_Σ + m + 2q - 2
    κ_A = ν_Σ + q
    Λ_Σ = 𝔼_inv_A
    for i in 1:m
        Λ_Σ += μ_γ[i] * μ_γ[i]' + Σ_γ[i]
    end
    𝔼_inv_Σ = κ_Σ * inv(Λ_Σ)
    Λ_A = diagm(diag(𝔼_inv_Σ)) + inv(ν_Σ * diagm(s²_Σ))
    𝔼_inv_A = κ_A * inv(Λ_A)
    return 𝔼_inv_Σ, 𝔼_inv_A
end