import os

from datetime import datetime, timedelta
from functools import reduce

from . import pre_process

exit()





def int_to_date(date):
	return datetime.strptime(str(date),"%Y%m%d")

def date_to_int(date):
#	print(date)
#	exit()
	return int(datetime.strftime(date,"%Y%m%d"))

def gen_dates(b_date, days):
    day = timedelta(days=1)
    print(b_date)
    print(type(b_date))
    print(day)
    print(type(day))
    exit()
    for i in range(days):
        yield b_date + day*i

def get_date_list(start, end):
    """
    获取日期列表
    """
    start = int_to_date(start)
    end = int_to_date(end)
    dates = []
    for d in gen_dates(start, (end-start).days + 1):
        dates.append(date_to_int(d))
    return dates

a = get_date_list(20170501, 20170611)

print(a)

exit()
import sys
from numpy import *
sys.path.append("return")
#import data_process as dp
import config
from MysqlConnection import MysqlConnection
from functools import reduce

a = arange(15).reshape(5,3)
print (a)
exit()

#from functools import reduce
#sql = "12354%s %s" % ('tt', 'pp')
#print (sql)
#exit()
#a = [1]
b = [0, 1, 2, 3,4 ,5, 6, 7, 8]
k = b[2:6]
print(k)



#c = a + b
#print (c)
#exit()
a = 10 **2
print(a)
exit()
def calculateChannelTotalReturn():
	total_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	raw_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.raw_dbname)
	sql = "select * from log_return_s_wja_1 where date = %s and channel_id = %s" % (20161217, 3)
	sql_ret = total_connection.query(sql)
	print (sql_ret)
	tmp = list(zip(*sql_ret))
	tmp[0] = [1]
	tmp [3] = [-2]
	print (tmp)
	new_data = list(map(sum, tmp))
	print (new_data)
	exit()
	new_data.append( reduce(lambda x,y : x + y, data) )
	for data in tmp:
		new_data.append( reduce(lambda x,y : x + y, data) )
	print (new_data)
	new_data = reduce(lambda x,y : x + y, new_data)
	print (new_data)
	exit()
	sql = "select * from log_return_s_wja_1 limit 10"
	sql_ret = total_connection.query(sql)
	date_channel_2_data = {}
	for data in sql_ret:
		date = data[0]
		channel = data[4]
		key = str(date) + '_' + str(channel)
		#print(key)
		if( key not in date_channel_2_data ):
			data[3] = -2
			date_channel_2_data[key] = data
		else:
			tmp = date_channel_2_data[key]
#			date_channel_2_data[1] += data[1]
			tmp[1] += data[1]
#			print (str(tmp[1])+ ' _ ' + str(date_channel_2_data[key][1]) )
			for i in range(5,34):
				tmp[i] += data[i]
#		print ('data[1] ' + str(date_channel_2_data[key][1]) )
	for key in date_channel_2_data.keys():
		data = date_channel_2_data[key]
#		print ( key  + ' : ' + str(data[1]) )
#		exit()
		for i in range(5,34):
#			print (str(data[i]) + " and " + str(data[2]) )
			data[i] = data[i] / data[2] * 100
		sql = "insert into log_return_s_wja_1_percent VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" % tuple(data)
		print(sql)
#		break
		#total_connection.query(sql)
calculateChannelTotalReturn()		
exit()

def all_dates():
	total_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	raw_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.raw_dbname)
	sql = "select date from log_return_s_wja_1"
	sql_ret = total_connection.query(sql)
	sql_ret = set(reduce(lambda x,y : x + y, sql_ret))
	print (sql_ret)
#all_dates()
#exit()
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
		print(temp)
		exit()
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
def calculateTotalReturn_twh():
	total_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	raw_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.raw_dbname)
	sql = "select * from log_return_s_wja_1 limit 10"
	sql_ret = total_connection.query(sql)
	date_2_info = {}
	for data in sql_ret:
		date = data[0]
		data[3] = -2
		data[4] =-2
#		print (data)
#		exit()
#		locale = data[3]
#		channel = data[4]
		if ( date not in date_2_info.keys() ):
			date_2_info[date] = data	
		else:
			date_2_info[date][1] += data[1]
			for i in range(5, 10) :
				date_2_info[date][i] += data[i]
#	print (date_2_info)
	for date in date_2_info.keys():
		info = date_2_info[date]
		for i in range(5, 10) :
				info[i] = info[i]/info[2] * 100
		print (info)
		sql = "inset into log_return_s_wja_1_percent values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" % tuple(info)
		print (sql)

calculateTotalReturn()
#all_dates()

exit()
total_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
raw_connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.raw_dbname)
#sql = "select * from log_return_s_wja_1 limit 3"
#result = raw_connection.query(sql)
#temp = list(zip(*result))
#print (temp)
#exit()
sql = "select date from log_return_s_wja_1 where 1 = 2 limit 3"

sql_ret = total_connection.query(sql)
sql_ret = list(zip(*sql_ret))
date_len = len(sql_ret)
if(date_len > 0):
	print (sql_ret[0])
else:
	print ("no ret")
exit()

sql = "select * from log_return_s_wja_1 limit 3"
sql_ret = total_connection.query(sql)
#sql_ret = sorted(list(set(reduce(lambda x,y : x + y, sql_ret))))
sql_ret = list(zip(*sql_ret))
trans = []
for data in sql_ret:
#	print (data)
	trans.append( reduce(lambda x,y : x + y, data))
print (trans)
exit()
print ("sql_ret:")
print ( sql_ret)
exit()
#sql_ret = list(zip(*sql_ret))
sql_ret = set(list(zip(*sql_ret)))
#for i in sql_ret:
#	print (i)
print (sql_ret)
exit()


x = [1, 2, 3]

y = [4, 5, 6]

z = [7, 8, 9]
xyz = zip(x, y, z)
for i in xyz:
	print (i)
a = dir(xyz)
print (a)
#tmp = zip([1], [2]) 
#print (tmp)
#tmp = zip(*sql_ret) 
#dates = dp.all_dates() 
#print (dates)
