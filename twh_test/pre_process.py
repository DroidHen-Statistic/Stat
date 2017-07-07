import os
import sys
from sys import argv
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(abs_path)
from utils import *

argc = len(argv)
if ( argc < 4 ):
	print ("params: date_start(YYYYMMDD), date_end(include), game_id, [server_id = 1]")
	exit()
date_start = argv[1]
date_end = argv[2]
game_id = argv[3]
server_id = -1
if ( argc > 4 ):
	server_id = argv[4]
log_type = 'item_used'

log_files = file_util.get_log_files(date_start, date_end, log_type, game_id, server_id)
if [ len(log_files) <= 0]:
    raise Exception('no file')
file_list = " ".join(log_files) 
#print(log_files)

tmp_dir = file_util.get_log_tmp_path(log_type, game_id, date_start, server_id)
out_file = os.path.join(tmp_dir, "after_read")
cmd = "cat %s | awk -v out_file=%s -f ./utils/pre_process.awk" % (file_list, out_file) 
#print(cmd)
os.system(cmd)
print("check file %s" % out_file)
#tmp_dir = file_util.get_log_tmp_path(log_type, game_id, date_start, server_id)
#awk_cmd = "awk -v out_dir=tmp_dir -f ../resource_get/util/awkFileSql %s"

#for log_file in log_files:
#   os.system(awk_cmd % log_file) 
