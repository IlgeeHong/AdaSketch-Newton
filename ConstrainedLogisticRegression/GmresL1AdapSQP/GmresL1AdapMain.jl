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

function GmresL1AdapMain(L1Adap, Prob)
    Verbose = L1Adap.verbose
    Max_Iter = L1Adap.MaxIter
    EPS_Res = L1Adap.EPS_Res
    mu = L1Adap.mu
    kappa = L1Adap.kappa
    eta = L1Adap.eta
    xi_B = L1Adap.xi_B
    nu = L1Adap.nu
    TotalRep = L1Adap.Rep
    GmresL1AdapR = Array{GmresL1AdapResult}(undef,length(Prob))
    # go over all Datasets
    for Idprob = 1:length(Prob)
        # load dataset
        println(Prob[Idprob])
        feature, label = load_dataset(string(Prob[Idprob]), dense = false, replace = false, verbose = true)
        N = size(feature)[1]
        nx = size(feature)[2]
        nlam = 11
        con_A = rand(Normal(0,1),(nlam-1, nx))
        con_b = rand(Normal(0,1),(nlam-1,))
        # define results vectors
        XStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        LamStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        KKTStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        alpha_Step = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        Grad_eval = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        Objcon_eval = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        TimeStep = reshape([[] for i = 1:TotalRep], (1,TotalRep))
        # go over all runs
        rep = 1
        while rep <= TotalRep
            println("GmresL1AdapSQP","-",Idprob,"-",rep)
            X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = GmresL1AdapSQP(feature,label,con_A,con_b,Max_Iter,EPS_Res,mu,kappa,eta,xi_B,nu)
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
    end
    return GmresL1AdapR
end
