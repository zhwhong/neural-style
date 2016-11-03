#!/usr/bin/env bash

dir=`dirname $0`
dir=`cd $dir; pwd`

style_dir="$dir/data/styles"
for i in "0 1 2"; do
    for s in `ls $style_dir`; do
        style="$dir/data/styles/$s"
        python neural_style.py --content-dir=$dir/data/hair --style-image=$style -i 30 --device='/gpu:3' --test-dir=$dir/data/hair-test
    done
done
