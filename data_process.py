from MysqlConnection import MysqlConnection
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import pandas as pd
from functools import reduce
import os

def calculateTotalReturn():
	"""
	统计每天总的留存，并以百分比的形式保存在新表中
	"""
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

def calculateChannelTotalReturn():
	"""
	统计不同channel_id的留存数据，并以百分比的形式保存在新表中
	"""
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
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



def level7DayLeft(levels):
	"""
	各个等级在不同日期的7日留存数
	
	按天统计各个等级7日留存，并以日期为横坐标，留存为纵坐标，每个等级画出一张图
	
	Arguments:
		levels {list} -- 要统计的等级，以列表形式给出
	"""
	sql = "select date,user_7day from log_level_left_s_wja_1 where level = %s"
	#sql = "select * from test"
	for level in levels:
		result = raw_connection.query(sql,level)
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
		#plt.savefig(os.path.dirname(__file__) + "/figures/7DayLeft_level/' + str(level) + ".jpg")  
		plt.cla()
		plt.show()

def date7DayLeft(dates):
	"""
	每日的各等级7日留存统计
	
	统计同一天的各个等级的7日留存数据，并以等级为横坐标，留存为纵坐标，每天画出一张图
	
	Arguments:
		dates {list} -- 要统计的日期，以列表形式给出
	"""
	sql = "select * from log_level_left_s_wja_1 where date = %s"
	for date in dates:
		print("------------",date,"--------------")
		result = raw_connection.query(sql,date)
		level = []
		number = []
		for record in result:
			print(record[1], record[2])
			level.append(record[1])	
			number.append(record[2])
		plt.plot(level,number,'ro-')
		plt.show()

def dateReturn(dates,channels = [-2]):
	"""
	某天注册的用户留存情况
	
	获取某一天注册的用户留存数据，并以天数为横坐标，留存率为纵坐标，每天做出一张图
	
	Arguments:
		dates {list} -- 要统计的日期，以列表形式给出
		channels {list} -- 要统计的channel_id,以列表形式给出，-2 为统计总和，默认为-2
	"""
	sql = "select * from log_return_s_wja_1_percent where date = %s and channel_id = %s"
	for channel in channels:
		path = os.path.abspath(os.path.dirname(__file__)) + "/figures/return_date/channe_" + str(channel)
		if not os.path.exists(path):
			os.mkdir(path)
		for date in dates:
			print("------------date:",date," channel:",channel,"--------------")
			result = total_connection.query(sql,[date,channel])
			if(len(result) != 0):
				number = result[0][5:34]
				days = range(2,31)
				plt.plot(days,number,'r--')
				plt.gca().set_xlabel('days')
				plt.gca().set_ylabel('return')
				plt.grid(True)
				plt.savefig(path + "/"+ str(date) + ".jpg")
				#plt.show()
				plt.cla()


def dayReturn(days):
	"""
	n-day留存情况
	
	获取不同日期下的同一个n-day 留存（2<=n<=30）,并以日期为横坐标，留存率为纵坐标，每一个n作出一张图
	
	Arguments:
		days {list} -- 要统计的n日留存，以列表方式给出
	"""
	for day in days:
		sql = "select date, "+ str(day) + "day from log_return_s_wja_1_percent where channel_id = %s"
		result = total_connection.query(sql,-2)
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
		#plt.savefig(os.path.dirname(__file__) + "/figures/return_day/" + str(day) + "day.jpg")
		#plt.show()
		plt.cla()

		fig.set_size_inches(10,5)
		plt.hist(number)
		plt.savefig(os.path.dirname(__file__) + "/figures/return_day/hist/" + str(day) + "day.jpg")
		#plt.show()

def levelTotal(start,end):
	"""
	画出每个等级的7日流失人数总和分布
	
	获取每个等级的7日流失用户总数，并以等级为横坐标，人数为纵坐标画图
	
	Arguments:
		start {int} -- 要统计的起始等级
		end {int} -- 要统计的结束等级
	"""
	sql = "select * from log_level_left_total"
	result = total_connection.query(sql)
	result = list(zip(*result))
	level = result[0][start:end]
	user_7day = result[1][start:end]
	plt.plot(level,user_7day)
	plt.gca().set_xlabel('level')
	plt.gca().set_ylabel('user_7day')
	plt.grid(True)
	#plt.savefig(os.path.dirname(__file__) + "/figures/7DayLeft_level_total/level_total_"+str(start) + "_" + str(end))
	plt.show()

def dauAnddnu():
	"""
	绘制每日活跃用户、新增用户以及两者差值（老用户）的曲线图
	
	"""
	sql = "select date, login_count, register_count from log_return_s_wja_1_percent where channel_id = %s"
	result = total_connection.query(sql,-2)
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
	#plt.savefig(os.path.dirname(__file__) + "/figures/dau and dnu.jpg",dpi=100)
	plt.show()

def dauAnddnu_Bar():
	"""
	dau和dnu的直方图
	"""
	sql = "select date, login_count, register_count from log_return_s_wja_1_percent where channel_id = %s"
	result = total_connection.query(sql,-2)
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
	plt.savefig(os.path.dirname(__file__) + "/figures/dau and dnu bar.jpg",dpi=100)
	plt.show()

if __name__ == '__main__':
	raw_connection = MysqlConnection("218.108.40.13","wja","wja","statistic")
	total_connection = MysqlConnection("218.108.40.13","wja","wja","wja")

	sql = "select date from log_return_s_wja_1_percent"
	result = total_connection.query(sql)
	dates = sorted(list(set(reduce(lambda x,y : x + y, result))))
	print(os.path.dirname(__file__))
	dateReturn(dates,[-1,3,66])

	raw_connection.close()
	total_connection.close()