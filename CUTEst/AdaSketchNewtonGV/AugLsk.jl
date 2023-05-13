function AugLsk(nlp,nx,x,lam,eta1,eta2,alpha,NewDir)
    x_sk = x+alpha*NewDir[1:nx]
    lam_sk = lam+alpha*NewDir[nx+1:end]
    f_sk, nabf_sk = objgrad(nlp,x_sk)
    c_sk, G_sk = consjac(nlp,x_sk)
    nab_xL_k = nabf_sk + G_sk'lam_sk
    AugL_sk = f_sk + c_sk'lam_sk + (eta1/2)*norm(c_sk)^2 + (eta2/2)*norm(nab_xL_k)^2
    return AugL_sk
end
