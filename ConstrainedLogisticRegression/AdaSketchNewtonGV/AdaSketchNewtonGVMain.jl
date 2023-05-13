include("AdaSketchNewtonGV.jl")

struct AdaSketchNewtonGVResult
    XStep::Array
    LamStep::Array
    KKTStep::Array
    alpha_Step::Array
    Grad_eval::Array
    Objcon_eval::Array
    TimeStep::Array
end

function AdaSketchNewtonGVMain(Aug, Prob)
    Verbose = Aug.verbose
    Max_Iter = Aug.MaxIter
    EPS_Res = Aug.EPS_Res
    eta1 = Aug.eta1
    eta2 = Aug.eta2
    delta = Aug.delta
    TotalRep = Aug.Rep
    xi_B = Aug.xi_B
    beta = Aug.beta
    nu = Aug.nu
    AdaSketchNewtonGVR = Array{AdaSketchNewtonGVResult}(undef,length(Prob))
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
            println("AdaSketch-Newton-GV","-",Idprob,"-",rep)
            X,Lam,KKT,Alpha,grad_eval,objcon_eval,Time,IdCon,IdSing = AdaSketchNewtonGV(feature,label,con_A,con_b,Max_Iter,EPS_Res,eta1,eta2,delta,xi_B,beta,nu)
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
        AdaSketchNewtonGVR[Idprob] = AdaSketchNewtonGVResult(XStep,LamStep,KKTStep,alpha_Step,Grad_eval,Objcon_eval,TimeStep)
    end
    return AdaSketchNewtonGVR
end
