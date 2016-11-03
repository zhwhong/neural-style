#!/usr/bin/env bash

dir=`dirname $0`
dir=`cd $dir;pwd`

PRISMA="$dir/data/prisma"

for style in `ls $PRISMA/*.jpg`; do
    for content in `ls $PRISMA/test/*.jpg`; do
        style_name=`basename $style .jpg`
        content_name=`basename $content`
        prisma="$PRISMA/$style_name/$content_name"
        if [ ! -f $prisma ]; then
            continue
        fi
        python prisma.py --s=$style --c=$content --p=$prisma >> prisma.loss 2>&1
    done
done
