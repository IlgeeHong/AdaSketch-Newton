# AdaSketch-Newton 

[AdaSketch-Newton](https://arxiv.org/pdf/2305.18379.pdf) <br>
Ilgee Hong, Sen Na, Michael Mahoney, Mladen Kolar <br>
[ICML 2023](https://icml.cc/Conferences/2023) <br>
<br>
**AdaSketch-Newton** is an algorithm for solving equality-constrained nonconvex optimization problems. The method uses **randomized iterative sketching** to compute a search direction and **exact augmented Lagrangian merit function** to evaluate a search direction and do line search. <br>
<br>
This repository provides a Julia implementation of AdaSketch-Newton methods and other baseline methods for the experiments in the paper. Specifically, the repository contains three folders for three problems in the paper (CUTEst, Constrained Logistic Regression, and PDE-constrained Problem).

## Setup

All our code is implemented in [Julia](https://julialang.org/) (ver 1.6.7). Install the packages that are used in the code, using <br>
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

## Usage

Run an experiment by executing
```julia
include("/file_location/problem_name/method_name/Main.jl")
```
* problem_name $\in$ {ConstrainedLogisticRegression, CUTEst, PDE} 
* method_name $\in$ {AdaSketchNewtonGV, AdaSketchNewtonRV, AL, GmresL1AdapSQP, GmresL1SQP}

You might need to edit the current working directory in Main.jl script.

## Citation
```
@InProceedings{hong2023constrained,
  title = {Constrained Optimization via Exact Augmented Lagrangian and Randomized Iterative Sketching},
  author = {Hong, Ilgee and Na, Sen and Mahoney, Michael W. and Kolar, Mladen},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {13174--13198},
  year = {2023}
}
```
