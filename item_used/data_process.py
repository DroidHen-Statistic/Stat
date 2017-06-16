import sys
sys.path.append("..")

import config
import os
from enum import Enum,unique
from MysqlConnection import MysqlConnection
import utils
import copy

@unique
class ItemUseFormat(Enum):
	UID = 0
	STAGE = 1
	ITEM = 2
	COUNT = 3
	DATA = 4
	FIRST = 5

#dir_list = [x for x in os.listdir(.)]F
#

def readLog(day_dir):
	"""
	读取一天的log文件，统计每个玩家各个道具的使用数量
	
	Arguments:
		day_dir {string} -- 该日的log文件夹路径
	
	Returns:
		dict -- 二维dict，{uid_1:{item_id_1:使用次数,...,item_id_n:使用次数},...,uid_n:{item_id_1:使用次数,...,item_id_n:使用次数}}
	"""
	files = os.listdir(day_dir)
	result = {}
	for file in files:
		#print(file)
		filename = os.path.join(day_dir,file)
		with open(filename,'r') as f:
			for line in f.readlines():
				line = line.split()
				uid = line[ItemUseFormat.UID.value]
				item = line[ItemUseFormat.ITEM.value]
				count = line[ItemUseFormat.COUNT.value]

				if not uid in result:
					result[uid] = {}
				item_id = "item_" + item
				if item_id in result[uid]:
					result[uid][item_id] += int(count)
				else:
					result[uid][item_id] = int(count)
	return result
					
	# 			sql = "select uid, " + item_id +" from " + table + " where uid = %s"
	# 			result = connection.query(sql,uid)
	# 			if len(result) == 0:
	# 				sql = "insert into " + table + " (uid, " + item_id +") values (%s,%s)"
	# 				connection.query(sql,[int(uid),1])
	# 			else:
	# 				sql = "update "+ table + " set " + item_id + " = " + item_id + " + 1 where uid = %s"
	# 				connection.query(sql,uid)
	# connection.close()

def calculateUserItemTable(years = -1, months = -1, days = -1):
	"""
	计算user_item表，元素为每个user使用各个item的次数
	
	Keyword Arguments:
		years {number} -- log日期——年，-1为全部 (default: {-1})
		months {number} -- log日期——月，-1为全部 (default: {-1})
		days {number} -- log日期——日，-1为全部 (default: {-1})
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	if years == -1:
		years = os.listdir(log_dir)
	for year in years:
		year_dir = os.path.join(log_dir,year)
		if months == -1:
			months = os.listdir(year_dir)
		for month in months:
			month_dir = os.path.join(year_dir, month)
			if days == -1:
				days = os.listdir(month_dir)
			for day in days:
				print("---------------" + day + "----------------")
				day_dir = os.path.join(month_dir,day)
				table = utils.item_user_table(101250)
				user_dict = readLog(day_dir)
				for uid in user_dict:
					sql = "select uid from " + table + " where uid = %s"
					result = connection.query(sql,uid)
					if len(result) == 0:
						keys = ""
						pattern = ""
						values = [int(uid)]
						for k,v in user_dict[uid].items():
							keys += (k + ', ')
							pattern +=("%s, ")
							values.append(v)
						sql = "insert into " + table + " (uid, " + keys[:-2] + ") values (%s," + pattern[:-2] + ")";
						print(sql)
						print(values)
						connection.query(sql,values)
					else:
						sql = "update " + table + " set "
						values = []
						for k,v in user_dict[uid].items():
							sql += (k + "= " + k + " + %s,")
							values.append(v)
						values.append(int(uid))
						sql = sql[0:-1] + " where uid = %s"
						#print(sql)
						connection.query(sql,values)
	connection.close()

	
if __name__ == "__main__":
	#print(sys.argv[1])
	log_dir = utils.log_dir("item_used",sys.argv[1])
	calculateUserItemTable()