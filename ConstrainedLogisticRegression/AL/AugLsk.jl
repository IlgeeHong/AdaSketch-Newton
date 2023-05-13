function AugLsk(feature,label,con_A,con_b,nx,x,lam,mu_k,alpha,NewDir)
    x_sk = x+alpha*NewDir[1:nx]
    f_sk = objec(feature, label, x_sk)
    c_sk = con(con_A, con_b, x_sk)
    AugL_sk = f_sk + lam'c_sk + (mu_k/2)*norm(c_sk)^2    
    return AugL_sk
end
