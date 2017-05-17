from MysqlConnection import MysqlConnection
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import pandas as pd
from functools import reduce
import os

def calculateTotalReturn():
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

def level7DayLeft(levels):
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
		#plt.savefig('E:/figures/7DayLeft_level/' + str(level) + ".jpg")  
		plt.cla()
		plt.show()

def date7DayLeft(dates):
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

def dateReturn(dates):
	sql = "select * from log_return_s_wja_1_percent where date = %s"
	for date in dates:
		print("------------",date,"--------------")
		result = total_connection.query(sql,date)
		number = result[0][5:34]
		days = range(2,31)
		plt.plot(days,number,'r--')
		plt.gca().set_xlabel('days')
		plt.gca().set_ylabel('return')
		plt.grid(True)
		#plt.savefig("E:/python/stat/figures/return_date/" + str(date) + ".jpg")
		plt.show()
		plt.cla()

def dayReturn(days):
	for day in days:
		sql = "select date, "+ str(day) + "day from log_return_s_wja_1_percent"
		result = total_connection.query(sql)
		result = list(zip(*result))
		number = result[1]
		s_number = sorted(number)
		with open(os.path.dirname(__file__) + '/' + str(day) + 'day_return.txt','w') as f:
			f.write(str(s_number))
		print(s_number)
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
		#plt.savefig("E:/python/stat/figures/return_day/" + str(day) + "day.jpg")
		#plt.show()
		plt.cla()

def levelTotal(start,end):
	sql = "select * from log_level_left_total"
	result = total_connection.query(sql)
	result = list(zip(*result))
	level = result[0][start:end]
	user_7day = result[1][start:end]
	plt.plot(level,user_7day)
	plt.gca().set_xlabel('level')
	plt.gca().set_ylabel('user_7day')
	plt.grid(True)
	#plt.savefig("E:/figures/7DayLeft_level_total/level_total_"+str(start) + "_" + str(end))
	plt.show()

def dauAnddnu():
	sql = "select date, login_count, register_count from log_return_s_wja_1_percent"
	result = total_connection.query(sql)
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
	#plt.savefig("E:/python/stat/figures/dau and dnu.jpg",dpi=100)
	plt.show()

def dauAnddnu_Bar():
	sql = "select date, login_count, register_count from log_return_s_wja_1_percent"
	result = total_connection.query(sql)
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
	plt.savefig("E:/python/stat/figures/dau and dnu bar.jpg",dpi=100)
	plt.show()

if __name__ == '__main__':
	raw_connection = MysqlConnection("218.108.40.13","wja","wja","statistic")
	total_connection = MysqlConnection("218.108.40.13","wja","wja","wja")

	sql = "select date from log_return_s_wja_1_percent"
	result = total_connection.query(sql)
	dates = sorted(list(set(reduce(lambda x,y : x + y, result))))

	dayReturn(range(2,31))

	raw_connection.close()
	total_connection.close()