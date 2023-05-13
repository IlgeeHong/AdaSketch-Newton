function L1Lsk(nlp,nx,x,mu,alpha,NewDir)
    x_sk = x+alpha*NewDir[1:nx]
    f_sk, _ = objgrad(nlp,x_sk)
    c_sk, _ = consjac(nlp,x_sk)
    L1L_sk = f_sk + mu*norm(c_sk,1)
    return L1L_sk
end
