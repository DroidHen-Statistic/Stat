import sys
sys.path.append("..")

import os
from enum import Enum,unique
from MysqlConnection import MysqlConnection
import utils

@unique
class ItemGetFormat(Enum):
	UID = 0
	STAGE = 1
	ITEM = 2
	COUNT = 3
	DATA = 4
	FIRST = 5
	VIA = 6

#dir_list = [x for x in os.listdir(.)]F
#

def readLog(day_dir, table):
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	files = os.listdir(day_dir)
	for file in files:
		filename = os.path.join(day_dir,file)
		with open(filename,'r') as f:
			for line in f.readlines():
				line = line.split()
				via = line[ItemGetFormat.VIA.value]
				if via == '1106' or via == '1107' or via == '1108':
					uid = line[ItemGetFormat.UID.value]
					item = line[ItemGetFormat.ITEM.value]
					item_id = "item_" + item
					sql = "select uid, " + item_id +" from " + table + " where uid = %s"
					result = connection.query(sql,uid)
					if len(result) == 0:
						sql = "insert into " + table + " (uid, " + item_id +") values (%s,%s)"
						connection.query(sql,[int(uid),1])
					else:
						sql = "update "+ table + " set " + item_id + " = " + item_id + " + 1 where uid = %s"
						connection.query(sql,uid)
	connection.close()

def calculateUserItemTable():
	years = os.listdir(log_dir)
	for year in years:
		year_dir = os.path.join(log_dir,year)
		months = os.listdir(year_dir)
		for month in months:
			month_dir = os.path.join(year_dir, month)
			days = os.listdir(month_dir)
			for day in days:
				day_dir = os.path.join(month_dir,day)
				table = utils.item_user_table(101917)
				readLog(day_dir,table)

if __name__ == "__main__":
	log_dir = utils.log_dir("item_get",101917)
	calculateUserItemTable()