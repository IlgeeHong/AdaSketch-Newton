include("AdaSketchNewtonRK.jl")

struct AdaSketchNewtonRKResult
    XStep::Array
    LamStep::Array
    KKTStep::Array
    alpha_Step::Array
    Grad_eval::Array
    Objcon_eval::Array
    TimeStep::Array
end

function AdaSketchNewtonRKMain(Aug, Prob)
    Verbose = Aug.verbose
    Max_Iter = Aug.MaxIter
    EPS_Res = Aug.EPS_Res
    eta1 = Aug.eta1
    eta2 = Aug.eta2
    delta = Aug.delta
    xi_B = Aug.xi_B
    beta = Aug.beta
    nu = Aug.nu
    TotalRep = Aug.Rep
    AdaSketchNewtonRKR = Array{AdaSketchNewtonRKResult}(undef,length(Prob))
    # go over all Problems
    for Idprob = 1:length(Prob)
        # load problem
        println(Prob[Idprob])
        nlp = CUTEstModel(Prob[Idprob])
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
            println("AdaSketch-Newton-RK","-",Idprob,"-",rep)
            X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = AdaSketchNewtonRK(nlp,Max_Iter,EPS_Res,eta1,eta2,delta,xi_B,beta,nu)
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
        AdaSketchNewtonRKR[Idprob] = AdaSketchNewtonRKResult(XStep,LamStep,KKTStep,alpha_Step,Grad_eval,Objcon_eval,TimeStep)
        finalize(nlp)
    end
    return AdaSketchNewtonRKR
end
