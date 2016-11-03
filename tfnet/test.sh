#!/usr/bin/env bash

dir=`dirname $0`
dir=`cd $dir; pwd`

style="$dir/data/styles/colormen3.jpg"

s=$1

for i in `ls $dir/data/hair-test`; do
    echo $i
    f="$dir/data/hair-test/$i"
    python stylize.py -i $f -o "$dir/output/${s}_$i" -s $s
done
