#!/usr/bin/env bash

dir=`dirname $0`
dir=`cd $dir; pwd`

GEN="g1"
MODEL_DIR=$dir/ckpt-dir

for meta in `find $MODEL_DIR -name "*.meta"`; do
    dirname=`dirname $meta`
    model_name=`basename $meta .meta`
    echo $model_name
    ls $dir/output/${model_name}* >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        continue
    fi
    for t in `ls $dir/data/hair-test`; do
        echo $t
        f="$dir/data/hair-test/$t"
        python stylize.py -i $f -o "$dir/output/${model_name}_$t" -g $GEN -m $dirname/$model_name --device="/gpu:0"
    done
done
