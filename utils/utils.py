import os
import config

from datetime import datetime, timedelta
from functools import reduce

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
	return "user_item_" + str(game_id)

def item_item_table(game_id):
	return "item_item_" + str(game_id)

def item_item_table(game_id):
	return "item_item_" + str(game_id)

def get_log_table(log_type, game_id, server_id = -1):
	return "log_" + log_type + "_s_wja_" + game_id +"_" + str(server_id)

# 2是api的版本号，目前是2
def get_log_type_path(log_type, game_id, server_id = -1):
	return os.path.join(config.log_base_dir,str(game_id),log_type +"_2")

def get_log_path(log_type, game_id, date, server_id = -1):
	year, month, day = split_date(date)
	if server_id == 1:
		ret = os.path.join(get_log_type_path(log_type, game_id), year, month, day)
	else:
		ret = os.path.join(get_log_type_path(log_type, game_id), str(server_id), year, month, day)
	return ret

def get_log_tmp_path(log_type, game_id, date, server_id = -1):
	return get_path(config.log_tmp_dir, str(game_id), log_type, str(date))

def get_log_type_tmp_path(log_type, game_id, server_id = -1):
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

# 返回文件夹列表
def get_log_dir_from_date(start, end, log_type, game_id, server_id=1):
    dates = get_date_list(start, end)
    dirs=[]
    for date in dates:
        dirs.append( get_log_path(log_type, game_id, date))
    return dirs

# 读入log，生成list结构
'''
def get_data_from_log(log_file):
    files = os.listdir(day_dir)
    for(cr_file in files):
        if():
            continue;
'''

def split_date(date):
	date_str = str(date)
	year = date_str[0:4]
	month = date_str[4:6]
	day = date_str[6:8]
	return year,month,day

