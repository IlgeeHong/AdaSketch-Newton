function AugLsk(nx,x,lam,eta1,eta2,alpha,NewDir,G_k,N)
    # AugL_sk
    x_sk = x+alpha*NewDir[1:nx]
    lam_sk = lam+alpha*NewDir[nx+1:end]
    # evaluate objective, gradient
    f_sk, nabf_sk = objandgrad(x_sk,N)
    # evaluate constraint and Jacobian
    c_sk = con(x_sk,N)
    G_sk = G_k
    # Lagrangian gradient
    nab_xL_k = nabf_sk + G_sk'lam_sk
    AugL_sk = f_sk + c_sk'lam_sk + (eta1/2)*norm(c_sk)^2 + (eta2/2)*norm(nab_xL_k)^2
    return AugL_sk
end
