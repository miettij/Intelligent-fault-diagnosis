#!/bin/bash

python3 main.py --dataset cwru --arch Ince --validate True --epochs 2 --lr 0.001 --batch-size 32 --bias False --DE-FE True
