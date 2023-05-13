function model_reduction(nabf_k,B_k,c_k,G_k,NewDir,mu,nx)
    m = -(nabf_k'*NewDir[1:nx])[1]-max((1/2)*NewDir[1:nx]'*B_k*NewDir[1:nx],0)+mu*(norm(c_k,1)-norm(G_k*NewDir[1:nx]+c_k,1))[1]
    return m
end
