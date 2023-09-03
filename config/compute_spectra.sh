#!/bin/bash

datasets=(ijcnn1 real-sim susy e2006 yearpredictionmsd yolanda)
k=100
destination=./spectra_results

for name in "${datasets[@]}"
do
    python compute_spectrum.py --data $name --k $k --dest $destination
done