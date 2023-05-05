#!/bin/bash

# Get the sequence and timestamp from input args
sequence=$1
timestamp=$2
mu_pos=$3
mu_after=$4
drugname=$5

python ./src/PubchemMapping.py $timestamp $drugname
python ./src/generateFeature.py $sequence $mu_pos $mu_after $timestamp
python ./src/prepareData.py $timestamp
python ./evaluation.py $timestamp




