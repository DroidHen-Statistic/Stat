import os
import config

from functools import reduce
from . import date_util

def get_path(base = config.base_dir, *paths):
	ret = base
	for path in paths:
		ret = os.path.join(ret,path)
		if not os.path.exists(ret):
			os.mkdir(ret)
	return ret

def get_figure_path(*subfolder):
	base = os.path.join(config.base_dir,"figures")
	return get_path(base,*subfolder)



def get_log_type_path(log_type, game_id, server_id = -1):
	return os.path.join(config.log_base_dir,game_id,log_type +"_2")

def get_log_path(log_type, game_id, date, server_id = -1):
	year, month, day = date_util.split_date(date)
	ret = os.path.join(get_log_type_path(log_type, game_id), year, month, day) if server_id == -1 else \
			os.path.join(get_log_type_path(log_type, game_id), str(server_id), year, month, day)
	return ret

def get_log_type_tmp_path(log_type, game_id, server_id = -1):
	return get_path(config.log_tmp_dir, str(game_id), log_type)

def get_log_tmp_path(log_type, game_id, date, server_id = -1):
	return get_path(config.log_tmp_dir, str(game_id), log_type, str(date))

def get_result_path(log_type, game_id, server_id = -1):
	return get_path(config.log_result_dir, str(game_id), log_type)

def item_used_total_file(game_id,date):
	log_tmp_path = get_log_tmp_path("item_used",game_id,date)
	return os.path.join(log_tmp_path, str(date) + "_total")

