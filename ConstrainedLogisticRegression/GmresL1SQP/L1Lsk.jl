function L1Lsk(feature,label,con_A,con_b,nx,x,mu,alpha,NewDir)
    x_sk = x+alpha*NewDir[1:nx]
    f_sk = objec(feature, label, x_sk)
    c_sk = con(con_A, con_b, x_sk)
    L1L_sk = f_sk + mu*norm(c_sk,1)
    return L1L_sk
end
