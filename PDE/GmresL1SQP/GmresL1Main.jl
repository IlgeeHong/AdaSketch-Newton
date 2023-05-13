include("GmresL1SQP.jl")

struct GmresL1Result
    XStep::Array
    LamStep::Array
    KKTStep::Array
    alpha_Step::Array
    Grad_eval::Array
    Objcon_eval::Array
    TimeStep::Array
end

function GmresL1Main(L1)
    Verbose = L1.verbose
    Max_Iter = L1.MaxIter
    EPS_Res = L1.EPS_Res
    mu = L1.mu
    kappa = L1.kappa
    kappa1 = L1.kappa1
    epsilon = L1.epsilon
    tau = L1.tau
    eta = L1.eta
    TotalRep = L1.Rep
    GmresL1R = Array{GmresL1Result}(undef,1)
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
        println("GmresL1SQP","-",rep)
        X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = GmresL1SQP(Max_Iter,EPS_Res,mu,kappa,kappa1,epsilon,tau,eta)
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
    GmresL1R[Idprob] = GmresL1Result(XStep,LamStep,KKTStep,alpha_Step,Grad_eval,Objcon_eval,TimeStep)
    
    return GmresL1R
end
