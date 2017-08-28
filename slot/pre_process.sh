#!/bin/bash

file=$1
#out_put="/tmp/after_process.txt"
#out_put="/home/wja/stat/Stat/slot/after_read"
#out_put="/home/wja/log/slot/after_read_04"
out_put=$2
#:> out_put
echo "begin_to_process: " $file
awk -f awkFile.awk $file >> $out_put
