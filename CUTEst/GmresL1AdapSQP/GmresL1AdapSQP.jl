include("L1Lsk.jl")
include("model_reduction.jl")

function GmresL1AdapSQP(nlp,Max_Iter,EPS_Res,mu,kappa,eta,xi_B,nu)
    # find n and m
    nx, nlam = nlp.meta.nvar, nlp.meta.ncon
    # initialize
    k, X, Lam, NewDir, grad_eval, objcon_eval, Alpha, Time = 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx+nlam), 0, 0, [], time()
    # indicator of convergence and singularity
    IdCon, IdSing = 1, 0
    # evaluate objective, gradient, Hessian
    # evaluate constraint and Jacobian
    f_k, nabf_k = objgrad(nlp, X[end])
    nab2f_k = hess(nlp, X[end])
    c_k, G_k = consjac(nlp, X[end])
    objcon_eval += 2
    grad_eval += 2
    # evaluate gradient and Hessian of Lagrangian
    nab_xL_k = nabf_k + G_k'Lam[end]
    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
    # KKT residual
    KKT = [norm([nab_xL_k; c_k])]
    # other parameters
    sigma = 0.1*(0.9)
    while KKT[end]>EPS_Res && k<Max_Iter
        # compute the least singular value squared of G_k
        xi_G = eigmin(Hermitian(Matrix(G_k*G_k'),:L))
        if xi_G < 1e-6
            IdSing = 1
            return [NaN],[NaN],[NaN],[NaN],NaN,NaN,NaN,0,IdSing
        else
            # compute the reduced Hessian and do Hessian modification
            Q_k, _ = qr(Matrix(G_k'))
            if nlam<nx && eigmin(Hermitian(Q_k[:,nlam+1:end]'nab_x2L_k*Q_k[:,nlam+1:end],:L)) < 1e-6
                 t_k = xi_B + norm(nab_x2L_k)
            else
                 t_k = 0
            end
            # generate B_k
            B_k = nab_x2L_k+t_k*Matrix(I,nx,nx)
            # build KKT system
            A, b = hcat(vcat(B_k,G_k),vcat(G_k',zeros(nlam,nlam))), -vcat(nab_xL_k,c_k)
            # initialize inexact direction and residual
            NewDir_t, r = zeros(nx+nlam), b
            # double While loops
            while true
                # evaluate threshold
                Quant2 = kappa*norm(b,1)
                t = 0
                while (norm(r,1) > Quant2) && (t < 1e5)
                    gmres!(NewDir_t,A,b,maxiter=2)
                    r = A*NewDir_t - b
                    t += 1
                end
                # model reduction
                model_reduc = model_reduction(nabf_k,B_k,c_k,G_k,NewDir_t,mu,nx)
                # model reduction threshold
                Quant4 = sigma*mu*max(norm(c_k,1),norm(G_k*NewDir_t[1:nx]+c_k,1)-norm(c_k,1))
                if mu >1e8
                    NewDir = NewDir_t
                    break
                elseif model_reduc < Quant4
                    mu *= nu
                    kappa *= (1/(nu)^2)
                else
                    NewDir = NewDir_t
                    break
                end
            end
            # directional derivative along inexact direction
            Quant1 = (nabf_k'NewDir[1:nx])[1]-mu*(norm(c_k,1)-norm(G_k*NewDir[1:nx]+c_k,1))[1]
            L1L_k = f_k + mu*norm(c_k,1)
            alpha_k = 1.0
            L1L_sk = L1Lsk(nlp,nx,X[end],mu,alpha_k,NewDir)
            objcon_eval += 2
            # Armijo condition
            while L1L_sk > L1L_k + alpha_k*eta*Quant1
                alpha_k *= 0.5
                L1L_sk = L1Lsk(nlp,nx,X[end],mu,alpha_k,NewDir)
                objcon_eval += 2
            end
            push!(X, X[end]+alpha_k*NewDir[1:nx])
            push!(Lam, Lam[end]+ alpha_k*NewDir[nx+1:end])
            push!(Alpha,alpha_k)
            k += 1
            # evaluate objective, gradient, Hessian
            # evaluate constraint and Jacobian
            f_k, nabf_k = objgrad(nlp, X[end])
            nab2f_k = hess(nlp, X[end])
            c_k, G_k = consjac(nlp, X[end])
            objcon_eval += 2
            grad_eval += 2
            # evaluate gradient and Hessian of Lagrangian
            nab_xL_k = nabf_k + G_k'Lam[end]
            nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
            push!(KKT, norm([nab_xL_k; c_k]))
            println(KKT[end])
        end
    end
    Time = time() - Time
    if k == Max_Iter
        return [NaN],[NaN],[NaN],[NaN],NaN,NaN,NaN,0,0
    else
        return X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,1,0
    end
end
