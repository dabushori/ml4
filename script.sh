#!/bin/bash
for i in {1..10}
do
    python3 ex4.py train_x train_y test_x "test_y${i}_byloss" > "output/out_${i}_byloss.txt"
done
