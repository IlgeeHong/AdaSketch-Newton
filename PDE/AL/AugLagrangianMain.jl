include("AugLagrangian.jl")

struct AugLagrangianResult
    XStep::Array
    LamStep::Array
    KKTStep::Array
    alpha_Step::Array
    Grad_eval::Array
    Objcon_eval::Array
    TimeStep::Array
end

function AugLagrangianMain(AugLag)
    Verbose = AugLag.verbose
    Max_Iter = AugLag.MaxIter
    EPS_Res = AugLag.EPS_Res
    mu = AugLag.mu
    beta = AugLag.beta
    TotalRep = AugLag.Rep
    AugLagrangianR = Array{AugLagrangianResult}(undef,1)
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
        println("AugLagrangian","-",rep)
        X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = AugLagrangian(Max_Iter,EPS_Res,mu,beta)
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
    AugLagrangianR[Idprob] = AugLagrangianResult(XStep,LamStep,KKTStep,alpha_Step,Grad_eval,Objcon_eval,TimeStep)
        
    
    return AugLagrangianR
end
