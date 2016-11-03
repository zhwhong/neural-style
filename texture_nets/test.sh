#!/bin/sh

dir=`dirname $0`
dir=`cd $dir; pwd`

val_dir="$dir/data/hair-test"
ckpt_dir="$dir/ckpt"

for model in `find $ckpt_dir -type f`; do
    echo "Model: $model"
    model_iter=`basename $model .t7`
    model_dir=`dirname $model`
    model_name=`basename $model_dir`
    model_dir=`dirname $model_dir`
    style=`basename $model_dir`

    output_dir="$dir/output/$style/$model_name/$model_iter"
    if [ ! -d $output_dir ]; then
        mkdir -p $output_dir
    fi
    if [ `ls $val_dir | wc -l` -eq `ls $output_dir | wc -l` ]; then
        continue
    fi
    for t in `ls $val_dir`; do
        echo $t
        th test.lua -input_image "$val_dir/$t" -save_path "$output_dir/$t" -model_t7 $model -cpu -image_size 512
    done
done

