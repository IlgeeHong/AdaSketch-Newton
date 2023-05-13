function con(A,b,x::AbstractVector{T}) where T
    c1 = A*x - b
    c2 = norm(x)^2-1
    return vcat(c1, c2)
end

function objec(X,y,x::Vector{T}) where T
    N = size(X)[1]
    temp = log.(ones(N,)+exp.(-y.*(X*x)))
    return mean(temp)
end

grad(X,y,x) = ForwardDiff.gradient(x -> objec(X,y,x), x)
Jac(A,b,x) = ForwardDiff.jacobian(x -> con(A,b,x), x)
Hess(X,y,x) = ForwardDiff.hessian(x -> objec(X,y,x), x)
