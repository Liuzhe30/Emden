#!/bin/bash

# Get the sequence and timestamp from input args
sequence=$1
timestamp=$2
mu_pos=$3
mu_after=$4
drugname=$5

python /data/emden/Emden-web/src/PubchemMapping.py $timestamp $drugname
python /data/emden/Emden-web/src/generateFeature.py $sequence $mu_pos $mu_after $timestamp
python /data/emden/Emden-web/src/prepareData.py $timestamp
python /data/emden/Emden-web/evaluation.py $timestamp




