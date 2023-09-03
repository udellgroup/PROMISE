#!/bin/bash

datasets=(susy)
rf_type=(gaussian)
m_rf=(20000)
b_rf=(1)

opts=(sgd saga)
opts_sketchy=(sketchysgd sketchysaga)
precond=(nystrom ssn)

problem_type=logistic
r_seed=1234
np_seed=2468
n_epochs=10
mu=0.01

destination=./streaming_results

for i in "${!datasets[@]}"
do
    name=${datasets[$i]}
    rf=${rf_type[$i]}
    m=${m_rf[$i]}
    b=${b_rf[$i]}
    for opt in "${opts[@]}"
    do
        python streaming_experiments.py --data $name --problem $problem_type --opt $opt --epochs $n_epochs --mu $mu  --rf_type $rf --m_rf $m --bandwidth_rf $b --dest $destination
        if [ $opt == "saga" ]
        then
            python streaming_experiments.py --data $name --problem $problem_type --opt $opt --epochs $n_epochs --mu $mu  --auto_lr --rf_type $rf --m_rf $m --bandwidth_rf $b --dest $destination
        fi
        
    done
    for opt in "${opts_sketchy[@]}"
    do
        for precond_type in "${precond[@]}"
        do
            python streaming_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --epochs $n_epochs --mu $mu  --rf_type $rf --m_rf $m --bandwidth_rf $b --dest $destination
        done
    done
done