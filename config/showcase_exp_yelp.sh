#!/bin/bash

dataset=yelp
time_budget=3600

opts=(svrg saga lkatyusha slbfgs)
opts_sketchy=(sketchysgd sketchysvrg sketchysaga sketchykatyusha)
precond=(nystrom sassn lessn ssn diagonal)

problem_type=logistic
r_seed=1234
np_seed=2468
n_epochs=100000000 # Make this large enough so that the time budget is the only stopping criterion
mu=0.01

destination=./showcase_results

for opt in "${opts[@]}"
do
    python showcase_experiments.py --data $dataset --problem $problem_type --opt $opt --time-budget $time_budget --epochs $n_epochs --mu $mu  --dest $destination
done
for opt in "${opts_sketchy[@]}"
do
    for precond_type in "${precond[@]}"
    do
        python showcase_experiments.py --data $dataset --problem $problem_type --opt $opt --precond $precond_type --time-budget $time_budget --epochs $n_epochs --mu $mu --dest $destination
    done
done