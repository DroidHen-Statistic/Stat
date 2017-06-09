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

def calculateTotalReturn():
	"""
	汇总各个渠道每天总的留存，并以百分比的形式保存在新表中
	"""

	raw_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.raw_dbname)
	total_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)

	sql = "select date from log_return_s_wja_1"
	result = raw_connection.query(sql)
	dates = []
	for record in result:
		dates.append(record[0])
	dates = sorted(list(set(dates)))
	for date in dates:
		sql = "select * from log_return_s_wja_1 where date = %s"
		result = raw_connection.query(sql,date)
		temp = list(zip(*result))
		temp[0] = [date]
		temp[3] = [-2]
		temp[4] = [-2]
		temp = list(map(sum,temp))
		for i in range(5,34):
			temp[i] = temp[i] / temp[2] * 100
		sql = "insert into log_return_s_wja_1_percent VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
		total_connection.query(sql,tuple(temp))

	raw_connection.close()
	total_connection.close()

def calculateChannelTotalReturn():
	"""
	统计不同channel_id的留存数据，并以百分比的形式保存在新表中
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	sql = "select date, channel_id from log_return_s_wja_1"
	result = connection.query(sql)
	result = list(zip(*result))
	dates = sorted(list(set(result[0])))
	channels = sorted(list(set(result[1])))

	for date in dates:
		for channel in channels:
			sql = "select * from log_return_s_wja_1 where date = %s and channel_id = %s"
			result = connection.query(sql,[date,channel])
			if(len(result)!=0):
				temp = list(zip(*result))
				temp[0] = [date]
				temp[3] = [-2]
				temp[4] = [channel]
				temp = list(map(sum,temp))
				if(temp[2] != 0):
					for i in range(5,34):
						temp[i] = temp[i] / temp[2] * 100
				print(date,"-----",channel)
				sql = "insert into log_return_s_wja_1_percent VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
				connection.query(sql,tuple(temp))
	connection.close()




def dateReturn(dates,channels = [-2]):
	"""
	某天注册的用户留存情况
	
	获取某一天注册的用户留存数据，并以天数为横坐标，留存率为纵坐标，每天做出一张图
	
	Arguments:
		dates {list} -- 要统计的日期，以列表形式给出
		channels {list} -- 要统计的channel_id,以列表形式给出，-2 为统计总和，默认为-2
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)

	sql = "select * from log_return_s_wja_1_percent where date = %s and channel_id = %s"
	for channel in channels:
		path = utils.figure_path("return_date_test", "channel_" + str(channel))
		for date in dates:
			print("------------date:",date," channel:",channel,"--------------")
			result = connection.query(sql,[date,channel])
			if(len(result) != 0):
				number = result[0][5:34]
				days = range(2,31)
				plt.plot(days,number,'r--')
				plt.gca().set_xlabel('days')
				plt.gca().set_ylabel('return')
				plt.grid(True)
				#plt.savefig(os.path.join(path,str(date)))
				plt.show()
				plt.cla()

	connection.close()

def dayReturn(days = list(range(2,31))):
	"""	
	n-day留存情况
	
	获取不同日期下的同一个n-day 留存（2<=n<=30）,并以日期为横坐标，留存率为纵坐标，每一个n作出一张图
	
	Keyword Arguments:
		days {list} -- 要统计的n日留存，以列表方式给出 (default: {list(range(2,31))})
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)

	for day in days:
		sql = "select date, "+ str(day) + "day from log_return_s_wja_1_percent where channel_id = %s"
		result = connection.query(sql,-2)
		result = list(zip(*result))
		number = result[1]
		# s_number = sorted(number)
		# with open(os.path.dirname(__file__) + '/' + str(day) + 'day_return.txt','w') as f:
		# 	f.write(str(s_number))
		# print(s_number)
		dates_num = result[0]
		dates = [str(x) for x in dates_num]
		fig = plt.gcf()
		fig.set_size_inches(18,9)
		plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
		plt.gca().xaxis.set_major_locator(mdate.AutoDateLocator())
		plt.xticks(pd.date_range(dates[0],dates[-1],freq='5d'))#时间间隔
		plt.xticks(rotation = 90)
		plt.plot(dates,number,'r-o',label = str(day) + "day return")
		plt.gca().set_xlabel('date')
		plt.gca().set_ylabel('return percent(%)')
		plt.legend(loc = "upper right")
		plt.grid(True)
		path = utils.figure_path("return_day_test")
		plt.savefig(os.path.join(path,str(day) + "day"))
		plt.show()
		plt.cla()

		fig.set_size_inches(10,5)
		plt.hist(number)
		path = utils.figure_path("return_day_test","hist")
		plt.savefig(os.path.join(path,str(day) + "day"))
		#plt.show()
	
	connection.close()




def dauAnddnu():
	"""
	绘制每日活跃用户、新增用户以及两者差值（老用户）的曲线图
	
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	sql = "select date, login_count, register_count from log_return_s_wja_1_percent where channel_id = %s"
	result = connection.query(sql,-2)
	result = list(zip(*result))
	dates = [str(x) for x in result[0]]
	plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
	plt.gca().xaxis.set_major_locator(mdate.AutoDateLocator())
	plt.xticks(pd.date_range(dates[0],dates[-1],freq='5d'))#时间间隔
	plt.xticks(rotation = 90)
	dau = result[1]
	dnu = result[2]
	old_user = []
	for i in range(len(dau)):
		old_user.append(dau[i] - dnu[i])
	plt.plot(dates,dau,'r-o',label = "dau")
	plt.plot(dates,dnu,'b-o',label = "dnu")
	plt.plot(dates,old_user,'g-o',label = "old_user")
	fig = plt.gcf()
	fig.set_size_inches(19,9)
	plt.gca().set_xlabel('date')
	plt.gca().set_ylabel('user')
	plt.legend(loc='upper right')
	path = utils.figure_path("dau and dnu test")
	plt.savefig(os.path.join(path,"dau and dnu"),dpi=100)
	plt.show()

	connection.close()

def dauAnddnu_Bar():
	"""
	dau和dnu的直方图
	"""

	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	sql = "select date, login_count, register_count from log_return_s_wja_1_percent where channel_id = %s"
	result = connection.query(sql,-2)
	result = list(zip(*result))
	dates = [str(x) for x in result[0]]
	fig = plt.gcf()
	fig.set_size_inches(18,9)
	plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
	plt.gca().xaxis.set_major_locator(mdate.AutoDateLocator())
	plt.xticks(pd.date_range(dates[0],dates[-1],freq='5d'))#时间间隔
	plt.xticks(rotation = 90)
	dau = result[1]
	dnu = result[2]
	old_user = []
	for i in range(len(dau)):
		old_user.append(dau[i] - dnu[i])
	plt.bar(dates,old_user,width = 0.35,label = "old_user")
	plt.bar(dates,dnu,bottom=old_user,width = 0.35, label = "dnu")
	plt.legend(loc = 'upper right')
	path = utils.figure_path("dau and dnu test")
	plt.savefig(os.path.join(path,"dau and dnu bar"),dpi=100)
	plt.show()

	connection.close()


def dnuOfChannelId(channels = [-2]):
	"""[summary]
	
	不同channel_id的dnu人数柱状图
	
	Keyword Arguments:
		channels {list} -- 要统计的channel_id 列表，-2为全部 (default: {[-2]})
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	sql = "select date, register_count from log_return_s_wja_1_percent where channel_id = %s"
	pre = channels[0]
	for channel in channels:

		result = connection.query(sql,channel)
		result = list(zip(*result))
		dates = [str(x) for x in result[0]]
		register_count = result[1]
		fig = plt.gcf()
		fig.set_size_inches(18,9)
		plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
		plt.gca().xaxis.set_major_locator(mdate.AutoDateLocator())
		plt.xticks(pd.date_range(dates[0],dates[-1],freq='5d'))#时间间隔
		plt.xticks(rotation = 90)
		if channel == pre:
			plt.bar(dates,register_count,width = 0.35,label = channel)
		else:
			plt.bar(dates,register_count,bottom = pre, width = 0.35,label = channel)
			pre = channel
	plt.legend(loc = 'upper right')
	path = utils.figure_path("dau and dnu")
	plt.savefig(os.path.join(path,"dnu of differente channel"))
	plt.show()
	connection.close()

def dnuOfChannelID_Percent(channels = [-1]):
	"""
	dnu不同channel占总数的比例曲线图
	
	Keyword Arguments:
		channels {list} -- 要统计的channel，以列表形式给出 (default: {[-1]})
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	sql = "select date,register_count from log_return_s_wja_1_percent where channel_id = %s"
	result = connection.query(sql,-2)
	result = list(zip(*result))
	dates = [str(x) for x in result[0]]
	total_register = result[1]
	sql = "select date,register_count from log_return_s_wja_1_percent where date = %s and channel_id = %s"
	for channel in channels:
		path = utils.figure_path()
		register_count = []
		for i in range(len(dates)):
			result = connection.query(sql,[dates[i],channel])
			if len(result):
				register_count.append(result[0][1]/total_register[i])
			else:
				register_count.append(0)
			fig = plt.gcf()
			fig.set_size_inches(18,9)
			plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
			plt.gca().xaxis.set_major_locator(mdate.AutoDateLocator())
			plt.xticks(pd.date_range(dates[0],dates[-1],freq='5d'))#时间间隔
			plt.xticks(rotation = 90)
		plt.plot(dates,register_count,label = channel)
		plt.legend(loc = 'upper right')
		plt.savefig(os.path.join(path,"dnu_percent_" + str(channel)),dpi=100)
		#plt.show()
		plt.cla()

	connection.close()

def all_dates():
	"""
	获取有记录的全部日期的列表
	
	Returns:
		list -- 全部的日期
	"""
	total_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)

	sql = "select date from log_return_s_wja_1_percent"
	result = total_connection.query(sql)
	dates = sorted(list(set(reduce(lambda x,y : x + y, result))))

	total_connection.close()
	return dates


if __name__ == '__main__':
	
	dates = all_dates()

	dnuOfChannelId()

	# raw_connection.close()
	# total_connection.close()