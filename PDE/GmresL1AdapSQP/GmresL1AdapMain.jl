include("GmresL1AdapSQP.jl")

struct GmresL1AdapResult
    XStep::Array
    LamStep::Array
    KKTStep::Array
    alpha_Step::Array
    Grad_eval::Array
    Objcon_eval::Array
    TimeStep::Array
end

function GmresL1AdapMain(L1Adap)
    Verbose = L1Adap.verbose
    Max_Iter = L1Adap.MaxIter
    EPS_Res = L1Adap.EPS_Res
    mu = L1Adap.mu
    kappa = L1Adap.kappa
    eta = L1Adap.eta
    nu = L1Adap.nu
    TotalRep = L1Adap.Rep
    GmresL1AdapR = Array{GmresL1AdapResult}(undef,1)
    # define results vectors
    XStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
    LamStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
    KKTStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
    alpha_Step = reshape([[] for i = 1:TotalRep], (1,TotalRep))
    Grad_eval = reshape([[] for i = 1:TotalRep], (1,TotalRep))
    Objcon_eval = reshape([[] for i = 1:TotalRep], (1,TotalRep))
    TimeStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
    # go over all runs
    Idprob = 1
    rep = 1
    while rep <= TotalRep
        println("GmresL1AdapSQP","-",rep)
        X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = GmresL1AdapSQP(Max_Iter,EPS_Res,mu,kappa,eta,nu)
        if IdSing == 1
            println("Singular")
            push!(XStep[rep], X)
            push!(LamStep[rep], Lam)
            push!(KKTStep[rep], KKT)
            push!(alpha_Step[rep], Alpha)
            push!(Grad_eval[rep], grad_eval)
            push!(Objcon_eval[rep], objcon_eval)
            push!(TimeStep[rep], Time)
            rep += 1
        elseif IdCon == 0
            println("Not convergent")
            push!(XStep[rep], X)
            push!(LamStep[rep], Lam)
            push!(KKTStep[rep], KKT)
            push!(alpha_Step[rep], Alpha)
            push!(Grad_eval[rep], grad_eval)
            push!(Objcon_eval[rep], objcon_eval)
            push!(TimeStep[rep], Time)
            rep += 1
        else
            push!(XStep[rep], X)
            push!(LamStep[rep], Lam)
            push!(KKTStep[rep], KKT)
            push!(alpha_Step[rep], Alpha)
            push!(Grad_eval[rep], grad_eval)
            push!(Objcon_eval[rep], objcon_eval)
            push!(TimeStep[rep], Time)
            rep += 1
        end
    end
    GmresL1AdapR[Idprob] = GmresL1AdapResult(XStep,LamStep,KKTStep,alpha_Step,Grad_eval,Objcon_eval,TimeStep)
    
    return GmresL1AdapR
end
