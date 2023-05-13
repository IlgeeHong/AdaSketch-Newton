function con(X,N)
    Z = reshape(X[(N^2+1):end],(N,N))
    W = reshape(X[1:N^2],(N,N))    
    row1 = zeros(1,N+2)
    row2 = zeros(1,N+2)
    col1 = zeros(N,)
    col2 = zeros(N,)
    W_extended = vcat(vcat(row1,hcat(hcat(col1,W),col2)),row2)
    cons = zeros(N,N)
    for i in 1:N
        for j in 1:N
            cons[i,j] = (W_extended[i,j+1] + W_extended[i+1,j] + W_extended[i+2,j+1] + W_extended[i+1,j+2] - 4*W_extended[i+1,j+1])/((2/(N+1))^2) + Z[i,j]
        end
    end                    
    c = reshape(cons,(N^2,))
    return c
end

function objandgrad(X,N)
    W_true = zeros(N,N)
    for i in 1:N
        for j in 1:N
            W_true[i,j] = sin((4 + ((1e-1)/sqrt(15))*(i-((N+1)/2)))) + cos((3 + ((1e-1)/sqrt(15))*(j-((N+1)/2))))
        end
    end     
    W_true_vec = reshape(W_true,(N^2,))
    obj = (1/2)*(norm(X[1:N^2]-W_true_vec)^2 + (1e-1)*norm(X[(N^2+1):end])^2)
    grad = zeros(2*N^2,)
    for i in 1:N^2
        grad[i] = X[i]-W_true_vec[i]
    end    
    for i in (N^2+1):(2*N^2)
        # change 1e-0 to 1
        grad[i] = (1e-1)*X[i]
    end    
    return obj, grad
end

function Hessian(N)
    # change 1e-0 to 1
    H = hcat(vcat(Matrix(I,N^2,N^2),zeros(N^2,N^2)),vcat(zeros(N^2,N^2),(1e-1)*Matrix(I,N^2,N^2)))
    return H
end

function Jacobian(N)
    temp = zeros(N+2,N+2)
    temp[2:(end-1),2:(end-1)] = (1/(2/(N+1))^2)*ones(N,N)
    jac1 = zeros(N^2,N^2)
    t = 1
    for m in 1:N^2
        M = m - (t-1)*N
        G = zeros(N+2,N+2)
        G[M,t+1] = temp[M,t+1]
        G[M+1,t] = temp[M+1,t]
        G[M+1,t+2] = temp[M+1,t+2]
        G[M+2,t+1] = temp[M+2,t+1]
        G_new = G[2:N+1,2:N+1]
        G_vec = reshape(G_new,(1,N^2))
        G_vec[m] = -4/(2/(N+1))^2
        jac1[m,:] = G_vec
        if m%N == 0
            t+=1
        end    
    end
    jac2 = Matrix(I,N^2,N^2)        
    jac = hcat(jac1,jac2)
    return jac
end

# G1 = [-4/(2*pi/3)^2 1/(2*pi/3)^2 1/(2*pi/3)^2 0 1 0 0 0]
#     G2 = [1/(2*pi/3)^2 -4/(2*pi/3)^2 0 1/(2*pi/3)^2 0 1 0 0]
#     G3 = [1/(2*pi/3)^2 0 1/(2*pi/3)^2 -4/(2*pi/3)^2 0 0 1 0]
#     G4 = [0 1/(2*pi/3)^2 1/(2*pi/3)^2 -4/(2*pi/3)^2 0 0 0 1]
#     G_k = vcat(G1,G2,G3,G4)

# -4 1 0 1 0 0 0 0 0
# 1 -4 1 0 1 0 0 0 0
# 0 1 -4 0 0 1 0 0 0
# 1 0 0 -4 1 0 1 0 0
# 0 1 0 1 -4 1 0 1 0
# 0 0 1 0 1 -4 0 0 1
