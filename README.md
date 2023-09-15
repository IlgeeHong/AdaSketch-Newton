# AdaSketch-Newton 

[AdaSketch-Newton](https://arxiv.org/pdf/2305.18379.pdf) <br>
Ilgee Hong, Sen Na, Michael Mahoney, Mladen Kolar <br>
[ICML 2023](https://icml.cc/Conferences/2023) <br>
<br>
**AdaSketch-Newton** is an algorithm for solving equality constrained nonconvex optimization problems. The method uses **randomized iterative sketching** to compute a search direction and **exact augmented Lagrangian merit function** to evaluate a search direction and do line search. <br>
<br>
This repository provides a Julia implementation of AdaSketch-Newton methods and other baseline methods for the experiments in the paper. Specifically, the repository contains three folders for three problems in the paper (CUTEst, Constrained Logistic Regression, PDE-constrained Problem).

## Setup

All our code is implemented in [Julia](https://julialang.org/). For installation of the packages that are used in the code, using <br>
```julia
pkg> add NLPModels
pkg> add LinearOperators
pkg> add OptimizationProblems
pkg> add MathProgBase
pkg> add ForwardDiff
pkg> add NLPModelsJuMP
pkg> add LinearAlgebra
pkg> add Distributed
pkg> add Ipopt
pkg> add DataFrames
pkg> add PyPlot
pkg> add MATLAB
pkg> add Glob
pkg> add DelimitedFiles
pkg> add Random
pkg> add Distributions
pkg> add Statistics
pkg> add IterativeSolvers
pkg> add CUTEst
pkg> add LIBSVMdata
```
