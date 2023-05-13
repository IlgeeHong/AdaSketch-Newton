include("AugLsk.jl")

function AugLagrangian(nlp,Max_Iter,EPS_Res,mu_k,beta)
    # find n and m
    nx, nlam = nlp.meta.nvar, nlp.meta.ncon
    # initialize
    k, X, Lam, NewDir, grad_eval, objcon_eval, Alpha, Time = 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx), 0, 0, [], time()
    tau_k = 0.1
    # indicator of convergence and singularity
    IdCon, IdSing = 1, 0
    # evaluate objective and gradient
    # evaluate constraint and Jacobian
    f_k, nabf_k = objgrad(nlp, X[end])
    c_k, G_k = consjac(nlp, X[end])
    objcon_eval += 2
    grad_eval += 2
    # evaluate gradient and Hessian of Lagrangian
    nab_xL_k = nabf_k + G_k'Lam[end]
    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
    # KKT residual
    KKT = [norm([nab_xL_k; c_k])]
    while KKT[end]>EPS_Res && k<Max_Iter
        # evaluate function, gradient and hessian of augmented Lagrangian
        AugL_k = f_k + Lam[end]'c_k + (mu_k/2)*norm(c_k)^2
        nab_AugL_k = nab_xL_k + mu_k*G_k'c_k
        hess_AugL_k = nab_x2L_k+(mu_k)*G_k'G_k+(mu_k)*(hess(nlp, X[end], obj_weight=1.0, c_k)-hess(nlp, X[end]))
        # construct modified hessian of augmented Lagrangian
        lamb = eigmin(Hermitian(Matrix(hess_AugL_k),:L))        
        mod_hess_AugL_k = hess_AugL_k + (1-lamb)*Matrix(I,nx,nx)
        # construct Newton system
        A, b = mod_hess_AugL_k, -nab_AugL_k
        NewDir_t, r = zeros(nx), -nab_AugL_k
        j = 0
        kappa = 1e-4
        # start augmented Lagrangian subproblem
        while norm(nab_AugL_k) > tau_k
            t = 0
            # find subproblem search direction using inexact Newton method with line search
            while (t < 1e5)
                gmres!(NewDir_t,A,b,maxiter=2)
                r = A*NewDir_t - b
                t += 1
                if norm(r) <= kappa*norm(b)
                    NewDir = NewDir_t
                    break
                end
            end
            Quant1 = (nab_AugL_k'NewDir)[1]
            alpha_k = 1.0
            AugL_sk = AugLsk(nlp,nx,X[end],Lam[end],mu_k,alpha_k,NewDir)
            objcon_eval += 2
            # Armijo condition
            while AugL_sk > AugL_k + alpha_k*beta*Quant1
                alpha_k *= 0.5
                AugL_sk = AugLsk(nlp,nx,X[end],Lam[end],mu_k,alpha_k,NewDir)
                objcon_eval += 2
            end
            # update subproblem iterate 
            push!(X, X[end]+alpha_k*NewDir)
            push!(Alpha,alpha_k)
            # evaluate objective and gradient
            # evaluate constraint and Jacobian
            f_k, nabf_k = objgrad(nlp, X[end])
            c_k, G_k = consjac(nlp, X[end])
            objcon_eval += 2
            grad_eval += 2
            # evaluate gradient and Hessian of Lagrangian
            nab_xL_k = nabf_k + G_k'Lam[end]
            nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
            # evaluate function, gradient and hessian of augmented Lagrangian
            AugL_k = f_k + Lam[end]'c_k + (mu_k/2)*norm(c_k)^2
            nab_AugL_k = nab_xL_k + mu_k*G_k'c_k
            hess_AugL_k = nab_x2L_k+(mu_k)*G_k'G_k+(mu_k)*(hess(nlp, X[end], obj_weight=1.0, c_k)-hess(nlp, X[end]))
            # construct modified hessian of augmented Lagrangian
            lamb = eigmin(Hermitian(Matrix(hess_AugL_k),:L))        
            mod_hess_AugL_k = hess_AugL_k + (1-lamb)*Matrix(I,nx,nx)
            # construct Newton system
            A, b = mod_hess_AugL_k, -nab_AugL_k
            j += 1
            if j > 1e2
                return [NaN],[NaN],[NaN],[NaN],NaN,NaN,NaN,0,0
            end    
        end
        # end augmented Lagrangian subproblem
        println(j)
        # update Lam    
        push!(Lam, Vector(Lam[end]+ mu_k*c_k))
        k += 1
        mu_k *= 1.5
        tau_k *= 0.1
        f_k, nabf_k = objgrad(nlp, X[end])
        c_k, G_k = consjac(nlp, X[end])
        objcon_eval += 2
        grad_eval += 2
        # evaluate gradient and Hessian of Lagrangian
        nab_xL_k = nabf_k + G_k'Lam[end]
        nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
        # KKT residual
        push!(KKT, norm([nab_xL_k; c_k]))
        println(KKT[end])
    end
    Time = time() - Time
    if k == Max_Iter
        return [NaN],[NaN],[NaN],[NaN],NaN,NaN,NaN,0,0
    else
        return X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,1,0
    end
end    