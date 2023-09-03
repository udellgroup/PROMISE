#!/bin/bash

datasets=(e2006 yearpredictionmsd santander jannis yolanda miniboone guillermo creditcard acsincome medical airlines click-prediction mtp elevators ailerons superconduct sarcos)

for name in "${datasets[@]}"
do
    python sklearn_opt.py --data $name --problem least_squares --iters 1000 --mu 0.1 --dest ./performance_results_mu_1e-1 
done