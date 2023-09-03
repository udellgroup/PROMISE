#!/bin/bash

datasets=(e2006 yearpredictionmsd santander jannis yolanda miniboone guillermo creditcard acsincome medical airlines click-prediction mtp elevators ailerons superconduct sarcos)

opts=(svrg saga lkatyusha slbfgs)
opts_sketchy=(sketchysvrg sketchysaga sketchykatyusha)
precond=(nystrom sassn lessn ssn diagonal)

problem_type=least_squares
r_seed=1234
np_seed=2468
n_epochs=100
mu=0.1

destination=./performance_results_mu_1e-1

csv_name=seed_${r_seed}_${np_seed}.csv

for name in "${datasets[@]}"
do
    for opt in "${opts[@]}"
    do
        if [ $opt == "saga" ]
        then
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $((n_epochs * 2)) --mu $mu --min_loc ./performance_results_mu_1e-1/$name/lsqr/$csv_name --dest $destination
        else
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $n_epochs --mu $mu --min_loc ./performance_results_mu_1e-1/$name/lsqr/$csv_name --dest $destination
        fi
    done
    for opt in "${opts_sketchy[@]}"
    do
        for precond_type in "${precond[@]}"
        do
            if [ $opt == "sketchysaga" ]
            then
                python performance_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --epochs $((n_epochs * 2)) --mu $mu --min_loc ./performance_results_mu_1e-1/$name/lsqr/$csv_name --dest $destination
            else
                python performance_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --epochs $n_epochs --mu $mu --min_loc ./performance_results_mu_1e-1/$name/lsqr/$csv_name --dest $destination
            fi
        done
    done
done