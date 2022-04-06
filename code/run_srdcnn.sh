#!/bin/bash

python3 main.py --dataset cwru --arch SRDCNN --validate True --epochs 2 --lr 0.0001 --batch-size 64 --bias True --DE-FE True
