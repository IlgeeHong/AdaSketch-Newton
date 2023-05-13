function AugLsk(feature,label,con_A,con_b,nx,x,lam,eta1,eta2,alpha,NewDir)
    # AugL_sk
    x_sk = x+alpha*NewDir[1:nx]
    lam_sk = lam+alpha*NewDir[nx+1:end]
    # evaluate objective, gradient
    f_sk = objec(feature, label, x_sk)
    nabf_sk = grad(feature, label, x_sk)
    # evaluate constraint and Jacobian
    c_sk = con(con_A, con_b, x_sk)
    G_sk = Jac(con_A, con_b, x_sk)
    # Lagrangian gradient
    nab_xL_k = nabf_sk + G_sk'lam_sk
    AugL_sk = f_sk + c_sk'lam_sk + (eta1/2)*norm(c_sk)^2 + (eta2/2)*norm(nab_xL_k)^2
    return AugL_sk
end
