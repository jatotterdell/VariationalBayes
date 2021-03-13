"""
    vec⁻¹(x)

Inverse of `vec` function.
Reshape a vector back into a square matrix.
"""
function vec⁻¹(V::AbstractVector{T})
    d = isqrt(length(V))
    return reshape(V, d, d)
end

"""
    vech(A)

Half vectorisation of a matrix.
"""
function vech(A::AbstractMatrix{T}) where T
    m = LinearAlgebra.checksquare(A)
    v = Vector{T}(undef, (m*(m+1))>>1)
    k = 0
    for j = 1:m, i = j:m
        @inbounds v[k += 1] = A[i,j]
    end
    return v
end

"""
    vech⁻¹(V)

Inverse of half-vectorisation of a matrix.
"""
function vech⁻¹(V::AbstractVector{T}) where T
    l = length(V)
    d = Integer(0.5*(sqrt(8*l + 1) - 1))
    @assert l == (d*(d+1))>>1
    A = Matrix{T}(undef, d, d)
    k = 0
    for j = 1:d, i = j:d
        @inbounds A[i,j] = V[k += 1]
        @inbounds A[j,i] = A[i,j]
    end
    return A
end

