#!/bin/bash

if [ $# -ne 4 ]; then
	echo "param: date_start, date_end, game_id, method"
	exit 1
fi

date_start=$1
date_end=$2
game_id=$3
method=$4

predir=`pwd`
cd $method
echo `pwd`

if [ -f get_raw_data.py ]; then
	python get_raw_data.py $date_start $date_end $game_id
fi

if [ -f pre_process.py ]; then
	python pre_process.py $date_start $date_end $game_id
fi

if [ -f process.py ]; then
	python process.py $game_id
fi

if [ -f output.py ]; then
	python output.py $game_id
fi


cd $predir