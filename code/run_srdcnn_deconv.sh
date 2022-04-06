#!/bin/bash

python3 main.py --dataset cwru --arch SRDCNN_deconv --validate True --epochs 2 --lr 0.1 --batch-size 128 --bias True --eps 1e-5 --stride 5 --freeze True --deconv-iter 5 --delinear True --delin-iter 5 --delin-eps 1e-5  --firstl-iter 25 --DE-FE True
