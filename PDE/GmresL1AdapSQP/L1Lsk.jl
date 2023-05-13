function L1Lsk(nx,x,mu,alpha,NewDir,N)
    x_sk = x+alpha*NewDir[1:nx]
    f_sk, _ = objandgrad(x_sk,N)
    c_sk = con(x_sk,N)
    L1L_sk = f_sk + mu*norm(c_sk,1)
    return L1L_sk
end
