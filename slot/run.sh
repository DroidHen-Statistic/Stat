#!bin/bash
#:>/home/wja/stat/Stat/slot/after_read
month=$1
out_put="/home/wja/log/slot/after_read_"$month
:>$out_put
for file in /home/wja/log/slot/2017/$month/*.log
do
    bash /home/wja/stat/Stat/slot/pre_process.sh $file $out_put
done
