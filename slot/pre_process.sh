#!/bin/bash

file=$1
#out_put="/tmp/after_process.txt"
out_put="/home/wja/stat/Stat/slot/after_read"
#:> out_put
awk -f awkFile.awk $file >> $out_put
