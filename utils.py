import os
import config

from datetime import datetime, timedelta
from functools import reduce
import types

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

def item_user_table(game_id):
	return "user_item_s_" + str(game_id)

def item_item_table(game_id):
	return "item_item_s_" + str(game_id)

def log_table(log_type, game_id):
	return "log_" + log_type + "_s_wja_" + gameid +"_1"

def get_log_type_path(log_type, game_id):
	return os.path.join(config.log_base_dir,"s_"+str(game_id),log_type +"_2")

def get_log_path(log_type, game_id, date):
	year, month, day = split_date(date)
	return os.path.join(get_log_type_path(log_type, game_id), year, month, day)

def get_log_tmp_path(log_type, game_id, date):
	return get_path(config.log_tmp_dir, str(game_id), log_type, str(date))

def get_log_type_tmp_path(log_type, game_id):
	return get_path(config.log_tmp_dir, str(game_id), log_type)

def item_used_total_file(game_id,date):
	log_tmp_path = get_log_tmp_path("item_used",game_id,date)
	return os.path.join(log_tmp_path, str(date) + "_total")

def union_dict(*objs,f = lambda x,y: x + y, initial = 0):
	"""
	合并多个字典，相同的键，值相加
	
	union_dict({'a':1, 'b':2, 'c':3}, {'a':2, 'b':3}) ----> {'a':3, 'b':5, 'c':3}
	
	Arguments:
		*objs {dict} -- 要合并的字典
	
	Returns:
		[dict] -- 合并后的字典
	"""
	keys = set(sum([list(obj.keys()) for obj in objs],[]))
	total = {}  
	for key in keys:
		total[key] = reduce(f,[obj.get(key,initial) for obj in objs])
	return total 

# def union_dict(*objs,f = lambda x,y: x + y, initial = 0):
# 	"""
# 	合并多个字典，相同的键，值相加
	
# 	union_dict({'a':1, 'b':2, 'c':3}, {'a':2, 'b':3}) ----> {'a':3, 'b':5, 'c':3}
	
# 	Arguments:
# 		*objs {dict} -- 要合并的字典
	
# 	Returns:
# 		[dict] -- 合并后的字典
# 	"""
# 	keys = set(sum([list(obj.keys()) for obj in objs],[]))
# 	total = {}  
# 	for key in keys: 
# 		total[key] = reduce(f,[obj.get(key,initial) for obj in objs])
# 	return total  

def date_to_int(date):
	return int(datetime.strftime(date,"%Y%m%d"))

def int_to_date(date):
	return datetime.strptime(str(date),"%Y%m%d")

def get_yesterday(date):
	return date_to_int(int_to_date(date) - timedelta(days = 1))

def gen_dates(b_date, days):
    day = timedelta(days=1)
    for i in range(days):
        yield b_date + day * i

def get_date_list(start, end):
    """
    获取日期列表
    """
    start = int_to_date(start)
    end = int_to_date(end)
    dates = []
    for d in gen_dates(start, (end-start).days + 1):
        dates.append(date_to_int(d))
    return dates

def split_date(date):
	date_str = str(date)
	year = date_str[0:4]
	month = date_str[4:6]
	day = date_str[6:8]
	return year,month,day

