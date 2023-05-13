include("AugLsk.jl")
include("Eval.jl")

function AdaSketchNewtonGV(Max_Iter,EPS_Res,eta1,eta2,delta,beta,nu)
    # data information
    N = 3
    nx, nlam = 2*N^2, N^2
    # initialize
    k, X, Lam, NewDir, grad_eval, objcon_eval, Alpha, Time = 0, [ones(nx,)], [ones(nlam,)], zeros(nx+nlam), 0, 0, [], time()
    # indicator of convergence and singularity
    IdCon, IdSing = 1, 0
    # evaluate objective, gradient, and Hessian
    # evaluate constraint and Jacobian
    f_k, nabf_k = objandgrad(X[end],N)
    nab2f_k = Hessian(N)
    c_k, G_k = con(X[end],N), Jacobian(N)
    objcon_eval += 2
    grad_eval += 2
    # evaluate gradient and Hessian of Lagrangian
    nab_xL_k = nabf_k + G_k'Lam[end]
    nab_x2L_k = nab2f_k
    # KKT residual and theta
    KKT = [norm([nab_xL_k; c_k])]
    theta = 1.0
    while KKT[end]>EPS_Res && k<Max_Iter
        # compute the least singular value squared of G_k
        xi_G = eigmin(Hermitian(Matrix(G_k*G_k'),:L))
        if xi_G < 1e-2
            IdSing = 1
            return [NaN],[NaN],[NaN],[NaN],NaN,NaN,NaN,0,IdSing
        else
            # generate B_k
            B_k = nab_x2L_k
            # compute Psi and Upsilon
            psi, Upsilon = 20*(max(norm(B_k)^2,1)/(min(xi_G,1)*(1e-1))), max(norm(G_k), norm(nab_x2L_k), 1)  
            # compute delta_trial
            delta_trial = ((0.5-beta)*eta2)/((1+eta1+eta2)*Upsilon^2*psi^2)
            # set delta
            delta = min(delta, delta_trial)
            # handle numerical issue
            if delta < 1e-6
                delta = 1e-6
            end
            # build KKT system
            A, b = hcat(vcat(B_k,G_k),vcat(G_k',zeros(nlam,nlam))), -vcat(nab_xL_k,c_k)
            # initialize inexact direction and residual
            NewDir_t, r = zeros(nx+nlam,), b
            # double While loops
            while true
                # compute accuracy threshold
                Quant2 = (theta*delta*norm(b))
                t = 0
                # adaptive accuracy condition
                Quant3 = norm(norm(A)*psi*r)
                while (Quant3 > Quant2) && (t < 1e6)
                    # Gaussian vector sketch
                    S = rand(Normal(0,1),(nlam+nx,)) 
                    NewDir_t -= ((S'*r)/(S'*A*A'*S))*(A'S)
                    r = A*NewDir_t - b
                    Quant3 = norm(norm(A)*psi*r)
                    t += 1
                end
                println(t)
                nabAugL_k = [nab_xL_k + eta2*nab2f_k*nab_xL_k + eta1*G_k'c_k; c_k+eta2*G_k*nab_xL_k]
                Quant1 = (nabAugL_k'NewDir_t)[1]
                # handle numerical issue
                if eta1 > 1e8 || eta2 <1e-8
                    NewDir = NewDir_t
                    break
                # descent direction condition    
                elseif Quant1 > -(eta2/2)*norm(b)^2
                    eta1 *= nu^2
                    eta2 *= (1/nu)
                    delta_trial = ((0.5-beta)*eta2)/((1+eta1+eta2)*Upsilon^2*psi^2)
                    delta = min((delta/nu^4), delta_trial)
                    # handle numerical issue
                    if delta < 1e-6
                        delta = 1e-6
                    end
                else
                    NewDir = NewDir_t
                    break
                end
            end
            # compute gradient of augmented Lagrangian
            nabAugL_k = [nab_xL_k + eta2*nab2f_k*nab_xL_k + eta1*G_k'c_k; c_k+eta2*G_k*nab_xL_k]
            Quant1 = (nabAugL_k'NewDir)[1]
            AugL_k = f_k + c_k'Lam[end] + (eta1/2)*norm(c_k)^2 + (eta2/2)*norm(nab_xL_k)^2
            alpha_k = 1.0
            AugL_sk = AugLsk(nx,X[end],Lam[end],eta1,eta2,alpha_k,NewDir,G_k,N)
            objcon_eval += 2
            grad_eval += 1
            # Armijo condition
            while AugL_sk > AugL_k + alpha_k*beta*Quant1
                alpha_k *= 0.5
                AugL_sk = AugLsk(nx,X[end],Lam[end],eta1,eta2,alpha_k,NewDir,G_k,N)
                objcon_eval += 2
                grad_eval += 1
            end
            push!(X, X[end]+alpha_k*NewDir[1:nx])
            push!(Lam, Lam[end]+ alpha_k*NewDir[nx+1:end])
            push!(Alpha,alpha_k)
            k += 1
            # evaluate objective and gradient
            # evaluate constraint and Jacobian
            f_k, nabf_k = objandgrad(X[end],N)
            c_k = con(X[end],N)
            grad_eval += 1
            objcon_eval += 2
            # evaluate gradient and Hessian of Lagrangian
            nab_xL_k = nabf_k + G_k'Lam[end]
            # update theta
            theta = 1.0 
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
