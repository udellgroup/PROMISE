#!/bin/bash

datasets=(ijcnn1 real-sim susy)

opts=(svrg saga lkatyusha slbfgs)
opts_sketchy=(sketchysgd sketchysvrg sketchysaga sketchykatyusha)
precond=(nystrom sassn lessn ssn diagonal)

problem_type=logistic
r_seed=1234
np_seed=2468
n_epochs=100
mu=0.01

destination=./suboptimality_results

for name in "${datasets[@]}"
do
    for opt in "${opts[@]}"
    do
        if [ $opt == "saga" ]
        then
            python suboptimality_experiments.py --data $name --problem $problem_type --opt $opt --epochs $((n_epochs * 2)) --mu $mu --dest $destination
        else
            python suboptimality_experiments.py --data $name --problem $problem_type --opt $opt --epochs $n_epochs --mu $mu --dest $destination
        fi
    done
    for opt in "${opts_sketchy[@]}"
    do
        for precond_type in "${precond[@]}"
        do
            if [ $opt == "sketchysgd" ] || [ $opt == "sketchysaga" ]
            then
                python suboptimality_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --epochs $((n_epochs * 2)) --mu $mu --dest $destination
            else
                python suboptimality_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --epochs $n_epochs --mu $mu --dest $destination
            fi
        done
    done
done