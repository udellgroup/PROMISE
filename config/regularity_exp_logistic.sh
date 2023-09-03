#!/bin/bash

datasets=(a9a rcv1 w8a real-sim ijcnn1 gisette mushrooms phishing)
opt_sketchy=sketchysaga
precond=nystrom
epochs=50
mu=0.01

dest=./regularity_results

for dataset in "${datasets[@]}"
do
    python regularity_experiments.py --data $dataset --opt $opt_sketchy --precond $precond --epochs $epochs --mu $mu --dest $dest
done
