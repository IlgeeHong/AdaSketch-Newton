using NLPModels
using LinearOperators
using OptimizationProblems
using MathProgBase
using ForwardDiff
using CUTEst
using NLPModelsJuMP
using LinearAlgebra
using Distributed
using Ipopt
using DataFrames
using PyPlot
using MATLAB
using Glob
using DelimitedFiles
using Random
using Distributions
using LIBSVMdata
using Statistics

cd("/.../ConstrainedLogisticRegression/AdaSketchNewtonGV")
# read problem
Prob = readdlm(string(pwd(),"/../Parameter/problems.txt"))

# define parameter module
module Parameter
    struct AugLagParams
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Res::Float64                   # Minimum KKT residual
        # adaptive parameters
        mu::Float64                        
        # fixed parameters  
        beta::Float64                                       
        Rep::Int                           # Number of Independent runs
    end
    struct AugParams
        verbose                            
        # stopping parameters
        MaxIter::Int                       
        EPS_Res::Float64                   
        # adaptive parameters
        eta1::Float64                      
        eta2::Float64                      
        delta::Float64                     
        # fixed parameters
        xi_B::Float64                      
        beta::Float64                      
        nu::Float64                        
        Rep::Int                           
    end
    struct L1Params
        verbose                            
        # stopping parameters
        MaxIter::Int                       
        EPS_Res::Float64                   
        # fixed parameters
        mu::Float64                        
        kappa::Float64                     
        kappa1::Float64                    
        epsilon::Float64                   
        tau::Float64                       
        eta::Float64                       
        xi_B::Float64                      
        Rep::Int
    end
    struct L1AdapParams
        verbose                            
        # stopping parameters
        MaxIter::Int                       
        EPS_Res::Float64                   
        # fixed parameters
        mu::Float64                        
        kappa::Float64                     
        eta::Float64                       
        xi_B::Float64                     
        nu::Float64                     
        Rep::Int
    end
end

using Main.Parameter
include("AdaSketchNewtonGVMain.jl")

#######################################
#########  run main file    ###########
#######################################
function main()
    Random.seed!(2023)
    include("../Parameter/Param.jl")
    AdaSketchNewtonGVR = AdaSketchNewtonGVMain(Aug, Prob)
    if Aug.verbose
        NumProb = 7
        decom = convert(Int64, floor(length(AdaSketchNewtonGVR)/NumProb))
        for i=1:decom
            path = string("../Solution/AdaSketchNewtonGV", i, ".mat")
            Result = AdaSketchNewtonGVR[(i-1)*NumProb+1:i*NumProb]
            write_matfile(path; Result)
        end
        path = string("../Solution/AdaSketchNewtonGVR", decom+1, ".mat")
        Result = AdaSketchNewtonGVR[decom*NumProb+1:end]
        write_matfile(path; Result)
    end
end

main()
