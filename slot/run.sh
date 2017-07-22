#!bin/bash
#:>/home/wja/stat/Stat/slot/after_read
for file in /home/wja/log/slot/2017/05/*.log
do
    echo $file
    bash /home/wja/stat/Stat/slot/pre_process.sh $file
done
