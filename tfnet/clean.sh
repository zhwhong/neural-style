./chwang.sh stop
rm -rf output/*
rm -rf ckpt-dir/*
rm -rf logs/*
rm -rf nohup.out
find . -name "*.pyc" | xargs rm
