#!/bin/bash

datasets=(epsilon susy a1a a2a a3a a4a a5a a6a a7a a8a covtype german.numer gisette ijcnn1 madelon mushrooms news20 phishing rcv1 real-sim splice sonar svmguide3 w1a w2a w3a w4a w5a w6a w7a webspam)
datasets2=(w8a a9a higgs)

for name in "${datasets[@]}"
do
    python sklearn_opt.py --data $name --problem logistic --iters 1000 --mu 0.01 --dest ./performance_results 
done
for name in "${datasets2[@]}"
do
    python sklearn_opt.py --data $name --problem logistic --iters 2000 --mu 0.01 --dest ./performance_results 
done