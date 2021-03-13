"""
    vec⁻¹(x)

Inverse of `vec` function.
Reshape a vector back into a square matrix.
"""
function vec⁻¹(x)
    d = Integer(sqrt(length(x)))
    return reshape(x, d, d)
end

function vech(x)

end

function vech⁻¹(x)

end