#!/bin/bash

datasets=(ijcnn1 real-sim susy)
n_runs_vec=(10 10 3)
opts_sketchy=(sketchysgd sketchysvrg sketchysaga sketchykatyusha)
precond=(nystrom)

freq_list=(0.5 1 2 5 100000) # Use 100000 to indicate no updates to the preconditioner after first iteration
rank_list=(1 2 5 10 20 50)

problem_type=logistic
r_seed=1234
np_seed=2468
n_epochs=40
mu=0.01

destination=./sensitivity_results

for i in "${!datasets[@]}"
do
    name=${datasets[$i]}
    n_runs=${n_runs_vec[$i]}
    for opt in "${opts_sketchy[@]}"
    do
        for precond_type in "${precond[@]}"
        do
            if [ $opt == "sketchysgd" ] || [ $opt == "sketchysaga" ]
            then
                python sensitivity_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --freq_list "${freq_list[@]}" --rank_list "${rank_list[@]}" --epochs $((n_epochs * 2)) --mu $mu --n_runs $n_runs --dest $destination
            else
                python sensitivity_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --freq_list "${freq_list[@]}" --rank_list "${rank_list[@]}" --epochs $n_epochs --mu $mu --n_runs $n_runs --dest $destination
            fi
        done
    done
done