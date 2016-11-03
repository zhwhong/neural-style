#!/usr/bin/env bash

dir=`dirname $0`
dir=`cd $dir; pwd`

style_dir="$dir/data/styles"
for i in `seq 0 10`; do
    for s in `ls $style_dir`; do
        style="$dir/data/styles/$s"
        python tflearn_train.py --contents=$dir/data/content --style=$style --epoches=30 --device='/gpu:0' --prefix=$i
    done
done
