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
using IterativeSolvers
using LIBSVMdata
using Statistics

cd("/.../ConstrainedLogisticRegression/GmresL1AdapSQP")
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
include("GmresL1AdapMain.jl")

#######################################
#########  run main file    ###########
#######################################
function main()
    Random.seed!(2023)
    include("../Parameter/Param.jl")
    GmresL1AdapR = GmresL1AdapMain(L1Adap, Prob)
    if L1Adap.verbose
        NumProb = 7
        decom = convert(Int64, floor(length(GmresL1AdapR)/NumProb))
        for i=1:decom
            path = string("../Solution/GmresL1AdapSQP", i, ".mat")
            Result = GmresL1AdapR[(i-1)*NumProb+1:i*NumProb]
            write_matfile(path; Result)
        end
        path = string("../Solution/GmresL1AdapR", decom+1, ".mat")
        Result = GmresL1AdapR[decom*NumProb+1:end]
        write_matfile(path; Result)
    end
end

main()
