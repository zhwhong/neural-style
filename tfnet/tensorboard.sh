#!/bin/sh

pid=`ps axf | grep tensorboard | grep 6007 | awk '{print $1}'`
if [ "$pid" != "" ]; then
    kill -9 $pid
fi
nohup tensorboard --logdir=./logs --port 6007 >/dev/null 2>&1 &
