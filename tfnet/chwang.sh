#!/usr/bin/env bash

self=$0
dir=`dirname $0`
dir=`cd $dir; pwd`

style_dir="$dir/data/styles"

function run() {
    learning_rate='0.001'
    for i in `seq 0 20`; do
        for s in `ls $style_dir`; do
            style="$dir/data/styles/$s"
            cmd="python chwang.py"
            cmd+=" --contents=$dir/data/content"
            cmd+=" --style=$style"
            cmd+=" --device=/gpu:1 --image_size=256 --batch_size=1"
            cmd+=" --learning_rate=$learning_rate"
            # cmd+=" --style_layers=relu4_2"
            # cmd+=" --generator=johnson"
            $cmd
        done
        learning_rate=`python -c "print $learning_rate * 0.7"`
    done
}

if [ "X$1" = "X" ]; then
    nohup bash $self run > nohup.out 2>&1 &
elif [ "X$1" = "Xstop" ]; then
    ps axf | grep 'chwang.sh' | grep -v 'stop' | awk '{print $1}' | xargs kill
    ps axf | grep 'chwang.py' | awk '{print $1}' | xargs kill
else
    run
    ./all_ckpt.sh
fi
