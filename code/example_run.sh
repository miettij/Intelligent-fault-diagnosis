#!/bin/bash

source env/bin/activate

python3 main.py --dataset cwru --arch wdcnn --deconv True --batchnorm False --delinear True --lr 0.0001 --batch-size 8 --deconv-iter 5 --eps 1e-5 --stride 5 --epochs 20
python3 main.py --dataset cwru --arch wdcnn --deconv False --batchnorm True --delinear False --lr 0.0001 --batch-size 8 --deconv-iter 5 --eps 1e-5 --stride 5 --epochs 20
