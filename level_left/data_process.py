import sys
sys.path.append("..")

import config
import utils
from MysqlConnection import MysqlConnection
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import pandas as pd
from functools import reduce
import os

def level7DayLeft(levels):
	"""
	各个等级在不同日期的7日留存数
	
	按天统计各个等级7日留存，并以日期为横坐标，留存为纵坐标，每个等级画出一张图
	
	Arguments:
		levels {list} -- 要统计的等级，以列表形式给出
	"""

	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)

	sql = "select date,user_7day from log_level_left_s_wja_1 where level = %s"
	#sql = "select * from test"
	for level in levels:
		result = connection.query(sql,level)
		dates_num = []
		number = []
		result = list(zip(*result))
		dates_num = result[0]
		number = result[1]
		dates = [str(x) for x in dates_num]
		plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
		plt.gca().xaxis.set_major_locator(mdate.AutoDateLocator())
		plt.xticks(pd.date_range(dates[0],dates[-1],freq='5d'))#时间间隔
		plt.xticks(rotation = 90)
		plt.plot(dates,number,'r--')
		path = utils.figure_path("7DayLeft_level")
		plt.savefig(os.path.join(path,str(level)))
		plt.show()
		plt.cla()
		
	connection.close()

def date7DayLeft(dates):
	"""
	每日的各等级7日留存统计
	
	统计同一天的各个等级的7日留存数据，并以等级为横坐标，留存为纵坐标，每天画出一张图
	
	Arguments:
		dates {list} -- 要统计的日期，以列表形式给出
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)

	sql = "select * from log_level_left_s_wja_1 where date = %s"
	for date in dates:
		print("------------",date,"--------------")
		result = connection.query(sql,date)
		level = []
		number = []
		for record in result:
			print(record[1], record[2])
			level.append(record[1])	
			number.append(record[2])
		plt.plot(level,number,'ro-')
		plt.show()

	connection.close()

def levelTotal(start,end):
	"""
	画出每个等级的7日流失人数总和分布
	
	获取每个等级的7日流失用户总数，并以等级为横坐标，人数为纵坐标画图
	
	Arguments:
		start {int} -- 要统计的起始等级
		end {int} -- 要统计的结束等级
	"""

	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	path = utils.figure_path("level_left")
	sql = "select * from log_level_left_total"
	result = connection.query(sql)
	result = list(zip(*result))
	level = result[0][start:end]
	user_7day = result[1][start:end]
	plt.grid(True)
	plt.bar(level,user_7day)
	plt.gca().set_xlabel('level')
	plt.gca().set_ylabel('user_7day')
	plt.savefig(os.path.join(path,"level_left_total_" + str(start) + "_" + str(end)))
	plt.show()

	connection.close()

def relativeLevelLeft(start,end):
	"""
	对应顶级范围的相对流失曲线
	
	该等级的流失人数除以前五个等级流失人数的平均数
	
	Arguments:
		start {int} -- 起始等级
		end {int} -- 结束等级
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)

	path = utils.figure_path("level_left")
	sql = "select * from log_level_left_total"
	result = connection.query(sql)
	result = list(zip(*result))
	level = result[0]
	user_7day = result[1]
	left_total = {}
	for i in range(len(level)):
		left_total[level[i]] = user_7day[i]
	relative_left = []
	for i in range(start,end + 1):
		pre = 0
		n = 0
		for j in range(i-5,i):
			if j in left_total:
				n += 1
				pre = pre + left_total[j]
		if pre != 0:
			relative_left.append(left_total[i] / (pre / n))
		else:
			relative_left.append(0)

	x = list(range(start,end + 1))
	plt.gca().set_xlabel('level')
	plt.gca().set_ylabel('relative left rate')
	plt.plot(x,relative_left)
	plt.grid(True)
	plt.savefig(os.path.join(path,"relative_level_left" + str(start) + "_" + str(end)))
	plt.show()

	connection.close()