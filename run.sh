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

python data_process.py $date_start $date_end $game_id
python algorithm.py $game_id
