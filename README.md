# AdaSketch-Newton 

[AdaSketch-Newton](https://arxiv.org/pdf/2305.18379.pdf) <br>
Ilgee Hong, Sen Na, Michael Mahoney, Mladen Kolar <br>
[ICML 2023](https://icml.cc/Conferences/2023) <br>
<br>
**AdaSketch-Newton** is an algorithm for solving equality constrained nonconvex optimization problems. The method uses **randomized iterative sketching** to compute a search direction and **exact augmented Lagrangian merit function** to evaluate a search direction and do line search. <br>
<br>
This repository provides a Julia implementation of AdaSketch-Newton methods and other baseline methods for the experiments in the paper "Constrained Optimization via Exact Augmented Lagrangian and Randomized Iterative Sketching".
