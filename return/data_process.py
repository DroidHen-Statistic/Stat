import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
import config
import utils
import pandas as pd
from MysqlConnection import MysqlConnection
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from functools import reduce
import platform

def calculateTotalReturn(date_start, date_end):
	"""
	汇总每天总的留存数据，并以百分比的形式保存在新表中
	
	Arguments:
		date_start {int} -- 要统计的开始日期
		date_end {int} -- 要统计的结束日期
	"""

	raw_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.raw_dbname)
	total_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)

	dates = utils.get_date_list(date_start,date_end)
	for date in dates:
		sql = "select * from log_return_s_wja_1 where date = %s"
		result = raw_connection.query(sql,date)
		temp = utils.union_dict(*result)
		# temp = list(zip(*result))
		temp['date'] = date
		temp['channel_id'] = -2
		temp['locale'] = -2
		# temp = list(map(sum,temp))
		values = [date,temp['login_count'],temp['register_count'],temp['locale'],temp['channel_id']]
		if temp['register_count'] != 0:
			for i in range(2,31):
				temp[str(i) +'day'] = temp[str(i) +'day'] / temp['register_count'] * 100
				values.append(temp[str(i) + 'day'])
		values.append(temp['login_count_pay'])
		sql = "delete from log_return_s_wja_1_percent where date = %s"
		total_connection.query(sql,date)
		sql = "insert into log_return_s_wja_1_percent (date, login_count,register_count,locale, channel_id"
		for i in range(2,31):
			sql += ", " + str(i) + "day"
		sql +=  ", login_count_pay) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
		total_connection.query(sql,values)

	raw_connection.close()
	total_connection.close()

def calculateChannelTotalReturn(date_start, date_end):
	"""
	统计不同channel_id的留存数据，并以百分比的形式保存在新表中
	
	Arguments:
		date_start {int} -- 要统计的开始日期
		date_end {int} -- 要统计的结束日期
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	sql = "select channel_id from log_return_s_wja_1"
	result = connection.query(sql)
	dates = utils.get_date_list(date_start,date_end)
	channels = set([x['channel_id'] for x in result])

	for date in dates:
		for channel in channels:
			sql = "select * from log_return_s_wja_1 where date = %s and channel_id = %s"
			result = connection.query(sql,[date,channel])
			if(len(result)!=0):
				temp = utils.union_dict(*result)
				# temp = list(zip(*result))
				temp['date'] = date
				temp['locale'] = -2
				temp['channel_id'] = channel
				# temp = list(map(sum,temp))
				values = [date,temp['login_count'],temp['register_count'],temp['locale'],temp['channel_id']]
				if temp['register_count'] != 0:
					for i in range(2,31):
						temp[str(i) +'day'] = temp[str(i) +'day'] / temp['register_count'] * 100
						values.append(temp[str(i) + 'day'])
				else:
					values += [0,] * 29
				values.append(temp['login_count_pay'])
				print(date,"-----",channel)
				sql = "delete from log_return_s_wja_1_percent where date = %s and channel_id = %s"
				connection.query(sql,(date,channel))
				sql = "insert into log_return_s_wja_1_percent (date, login_count,register_count,locale, channel_id"
				for i in range(2,31):
					sql += ", " + str(i) + "day"
				sql +=  ", login_count_pay) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
				# print(sql)
				# print(values)
				connection.query(sql,values)
	connection.close()


def dateReturn(date_start, date_end, channels = [-2]):
	"""
	某天注册的用户留存情况
	
	获取某一天注册的用户留存数据，并以天数为横坐标，留存率为纵坐标，每天做出一张图
	
	
	Arguments:
		date_start {int} -- 要统计的开始日期
		date_end {int} -- 要统计的结束日期
	
	Keyword Arguments:
		channels {list} -- 要统计的channel_id,以列表形式给出，-2 为统计总和 (default: {[-2]})
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	dates = utils.get_date_list(date_start,date_end)
	sql = "select * from log_return_s_wja_1_percent where date = %s and channel_id = %s"
	for channel in channels:
		path = utils.get_figure_path("return_date_test", "channel_" + str(channel))
		print(path)
		for date in dates:
			print("------------date:",date," channel:",channel,"--------------")
			result = connection.query(sql,[date,channel])
			if(len(result) != 0):
				return_percent = []
				for i in range(2,31):
					return_percent.append(result[0][str(i) + 'day'])
				# number = result[0][5:34]
				days = range(2,31)
				plt.plot(days,return_percent,'r--')
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
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)

	for day in days:
		sql = "select date, "+ str(day) + "day from log_return_s_wja_1_percent where channel_id = %s"
		result = connection.query(sql,-2)
		# result = list(zip(*result))
		# number = result[1]
		# dates_num = result[0]
		return_percent = [x[str(day) + 'day'] for x in result]
		dates= [str(x['date']) for x in result]

		#dates = [str(x) for x in dates_num]
		fig = plt.gcf()
		fig.set_size_inches(18,9)
		plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
		plt.gca().xaxis.set_major_locator(mdate.AutoDateLocator())
		plt.xticks(pd.date_range(dates[0],dates[-1],freq='5d'))#时间间隔
		plt.xticks(rotation = 90)
		plt.plot(dates,return_percent,'r-o',label = str(day) + "day return")
		plt.gca().set_xlabel('date')
		plt.gca().set_ylabel('return percent(%)')
		plt.legend(loc = "upper right")
		plt.grid(True)
		path = utils.get_figure_path("return_day_test")
		plt.savefig(os.path.join(path,str(day) + "day"))
		plt.show()
		plt.cla()

		fig.set_size_inches(10,5)
		plt.hist(return_percent)
		path = utils.get_figure_path("return_day_test","hist")
		plt.savefig(os.path.join(path,str(day) + "day"))
		plt.show()
	
	connection.close()




def dauAndDnu():
	"""
	绘制每日活跃用户、新增用户以及两者差值（老用户）的曲线图
	
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	sql = "select date, login_count, register_count from log_return_s_wja_1_percent where channel_id = %s"
	result = connection.query(sql,-2)
	#result = list(zip(*result))
	dates = [str(x['date']) for x in result]
	#dates = [str(x) for x in result[0]]
	plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
	plt.gca().xaxis.set_major_locator(mdate.AutoDateLocator())
	plt.xticks(pd.date_range(dates[0],dates[-1],freq='5d'))#时间间隔
	plt.xticks(rotation = 90)
	# dau = result[1]
	# dnu = result[2]
	dau = [x['login_count'] for x in result]
	dnu = [x['register_count'] for x in result]
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
	path = utils.get_figure_path("dau and dnu test")
	#plt.savefig(os.path.join(path,"dau and dnu"),dpi=100)
	plt.show()

	connection.close()

def dauAndDnu_Bar():
	"""
	dau和dnu的直方图
	"""

	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	sql = "select date, login_count, register_count from log_return_s_wja_1_percent where channel_id = %s"
	result = connection.query(sql,-2)
	# result = list(zip(*result))
	dates = [str(x['date']) for x in result]
	dau = [x['login_count'] for x in result]
	dnu = [x['register_count'] for x in result]
	fig = plt.gcf()
	fig.set_size_inches(18,9)
	plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
	plt.gca().xaxis.set_major_locator(mdate.AutoDateLocator())
	plt.xticks(pd.date_range(dates[0],dates[-1],freq='5d'))#时间间隔
	plt.xticks(rotation = 90)

	old_user = []
	for i in range(len(dau)):
		old_user.append(dau[i] - dnu[i])
	plt.bar(dates,old_user,width = 0.35,label = "old_user")
	plt.bar(dates,dnu,bottom=old_user,width = 0.35, label = "dnu")
	plt.legend(loc = 'upper right')
	path = utils.get_figure_path("dau and dnu test")
	plt.savefig(os.path.join(path,"dau and dnu bar"),dpi=100)
	plt.show()

	connection.close()


def dnuOfChannelId(channels = [-2]):
	"""
	
	不同channel_id的dnu人数柱状图
	
	Keyword Arguments:
		channels {list} -- 要统计的channel_id 列表，-2为全部 (default: {[-2]})
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	sql = "select date, register_count from log_return_s_wja_1_percent where channel_id = %s"
	pre = channels[0]
	for channel in channels:
		result = connection.query(sql,channel)
		# result = list(zip(*result))
		dates = [str(x['date']) for x in result]
		register_count = [x['register_count'] for x in result]
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
	path = utils.get_figure_path("dau and dnu")
	#plt.savefig(os.path.join(path,"dnu of differente channel"))
	plt.show()
	connection.close()

def dnuOfChannelID_Percent(channels = [-1]):
	"""
	dnu不同channel占总数的比例曲线图
	
	Keyword Arguments:
		channels {list} -- 要统计的channel，以列表形式给出 (default: {[-1]})
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	sql = "select date,register_count from log_return_s_wja_1_percent where channel_id = %s"
	result = connection.query(sql,-2)
	# result = list(zip(*result))
	dates = [str(x['date']) for x in result]
	total_register = [x['register_count'] for x in result]
	# total_register = result[1]
	sql = "select date,register_count from log_return_s_wja_1_percent where date = %s and channel_id = %s"
	for channel in channels:
		path = utils.get_figure_path()
		register_count = []
		for i in range(len(dates)):
			result = connection.query(sql,[dates[i],channel])
			if len(result):
				register_count.append(result[0]['register_count']/total_register[i])
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
		#plt.savefig(os.path.join(path,"dnu_percent_" + str(channel)),dpi=100)
		plt.show()
		plt.cla()

	connection.close()

# def all_dates():
# 	"""
# 	获取有记录的全部日期的列表
	
# 	Returns:
# 		list -- 全部的日期
# 	"""
# 	total_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)

# 	sql = "select date from log_return_s_wja_1_percent"
# 	result = total_connection.query(sql)
# 	result = [x['date'] for x in result]
# 	dates = sorted(list(set(result)))

# 	total_connection.close()
# 	return dates


if __name__ == '__main__':
	if platform.system() == "Linux":
		import matplotlib
		matplotlib.use('agg')
	# dates = all_dates()

	calculateChannelTotalReturn(20161216,20170423)

	# raw_connection.close()
	# total_connection.close()