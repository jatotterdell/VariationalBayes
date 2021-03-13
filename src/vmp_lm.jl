"""
    vmp_lm(X, y, μ₀, Σ₀, a, maxiter = 100, tol = 1e-8, verbose = true)
"""
function vmp_lm(X, y, μ₀, Σ₀, a, maxiter = 100, tol = 1e-8, verbose = true)
    n = length(y)
    d = size(X)[2]
    G = diagm(ones(1))
    ξ = 1.0
    Λ = diagm(ones(1)) / a^2

    # Prior fragment parameters
    ηpβ = GaussianPriorFragment(μ₀, Σ₀)
    ηpa = InverseWishartPriorFragment(ξ, Λ)
    ηpy = [GaussianPriorFragment(ones(d), diagm(ones(d))), [-2.0; -1.0]]
    ηpσ = [[-2.0; -1.0], [-2.0; -1.0]]

    # Sufficient statistics
    XtX = X'X
    Xty = X'y
    yty = y'y

    # Variational parameters
    ηβ = ηpβ + ηpy[1]
    ησ = ηpσ[1] + ηpy[2]
    ηa = ηpa + ηpσ[2]

    i = 1
    converged = false
    elbo = zeros(maxiter)
    while(i ≤ maxiter && !converged)
        ηpy = GaussianLikelihoodFragment(n, XtX, Xty, yty, ηβ, ησ)
        ηpσ = IteratedInverseGWishartFragment(G, ξ, ησ, ηa);
        ηβ = ηpβ + ηpy[1]
        ησ = ηpσ[1] + ηpy[2]
        ηa = ηpa + ηpσ[2]
        i += 1 
    end

    θβ = GaussianCommonParameters(ηβ)
    θσ = InverseGWishartCommonParameters(ησ)
    θa = InverseGWishartCommonParameters(ηa)
    return Dict([
        ("Iterations", i), 
        ("β", MvNormal(θβ[1], θβ[2])),
        ("σ", InverseWishart(θσ[1], θσ[2])),
        ("a", InverseWishart(θa[1], θa[2]))
    ])
end
