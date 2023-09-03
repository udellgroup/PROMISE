#!/bin/bash

datasets=(a1a a2a a3a a4a a5a a6a a7a a8a a9a covtype german.numer gisette ijcnn1 madelon mushrooms news20 phishing rcv1 real-sim splice sonar svmguide3 w1a w2a w3a w4a w5a w6a w7a w8a webspam epsilon higgs susy)

opts=(svrg saga lkatyusha slbfgs)
opts_sketchy=(sketchysvrg sketchysaga sketchykatyusha)
precond=(nystrom sassn lessn ssn diagonal)

problem_type=logistic
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
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $((n_epochs * 2)) --mu $mu --min_loc ./performance_results_mu_1e-1/$name/lbfgs/$csv_name --dest $destination
        else
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $n_epochs --mu $mu --min_loc ./performance_results_mu_1e-1/$name/lbfgs/$csv_name --dest $destination
        fi
    done
    for opt in "${opts_sketchy[@]}"
    do
        for precond_type in "${precond[@]}"
        do
            if [ $opt == "sketchysaga" ]
            then
                python performance_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --epochs $((n_epochs * 2)) --mu $mu --min_loc ./performance_results_mu_1e-1/$name/lbfgs/$csv_name --dest $destination
            else
                python performance_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --epochs $n_epochs --mu $mu --min_loc ./performance_results_mu_1e-1/$name/lbfgs/$csv_name --dest $destination
            fi
        done
    done
done