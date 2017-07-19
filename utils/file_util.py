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

# 返回文件夹列表
def get_log_dir_from_date(start, end, log_type, game_id, server_id = -1):
    dates = date_util.get_date_list(start, end)
    dirs=[]
    for date in dates:
        dirs.append( get_log_path(log_type, game_id, date))
    return dirs

# 返回log文件列表
def get_log_files(date_start, date_end, log_type, game_id, server_id = -1):
    log_files = []
    log_file_dirs = get_log_dir_from_date(date_start, date_end, log_type, game_id, server_id)
    for log_dir in log_file_dirs:
        if (os.path.isdir(log_dir)) == False:
#            print("%s no log" % log_dir)
            continue
        files = os.listdir(log_dir)
        for log_file in files:
            full_file = os.path.join(log_dir, log_file) 
            if os.path.isfile(full_file):
                log_files.append(full_file)
#                read_log_file(full_file, uid_item_count)
        return log_files
