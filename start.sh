#!/bin/bash

for seed in {0..9}
do
    for algorithm in filtered generic
    do
        mkdir -p results/python3/$algorithm/$seed
        python3 -B run.py $algorithm $seed > results/python3/$algorithm/$seed/result.csv
        mkdir -p results/pypy3/$algorithm/$seed
        pypy3 -B run.py $algorithm $seed > results/pypy3/$algorithm/$seed/result.csv
    done
done

python3 -B plot.py results
