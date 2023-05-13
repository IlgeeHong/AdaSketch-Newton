function AugLsk(nx,x,lam,mu_k,alpha,NewDir,N)
    x_sk = x+alpha*NewDir[1:nx]
    f_sk, _ = objandgrad(x_sk,N)
    c_sk = con(x_sk,N)
    AugL_sk = f_sk + lam'c_sk + (mu_k/2)*norm(c_sk)^2    
    return AugL_sk
end
