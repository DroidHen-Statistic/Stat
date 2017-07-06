import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils import *
from MysqlConnection import MysqlConnection
import pickle




game_id = sys.argv[1]
connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
result_path = file_util.get_result_path("item_used", game_id)
with open(os.path.join(result_path,"result"),'rb') as f:
	item_item_relation = pickle.load(f)

item_item_table = db_util.get_item_item_table(game_id)

for k,v in item_item_relation.items():
	for vk, vv in v.items():
		sql = "update " + item_item_table + " set " + vk + " = %s where item_id = %s"
		connection.query(sql,[float(vv),k[5:]])

