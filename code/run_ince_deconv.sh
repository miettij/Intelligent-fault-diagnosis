#!/bin/bash

python3 main.py --dataset cwru --arch Ince_deconv --validate True --epochs 2 --lr 0.0001 --batch-size 8 --bias True --eps 1e-5 --stride 5 --freeze True --deconv-iter 5 --delinear True --delin-iter 5 --delin-eps 1e-5  --firstl-iter 25 --DE-FE True
