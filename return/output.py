import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from MysqlConnection import MysqlConnection
from utils import *

def dateReturn(date_start, date_end, game_id, channels = [-2], locales = [-2]):
	"""
	某天注册的用户留存情况
	
	获取某一天注册的用户留存数据，并以天数为横坐标，留存率为纵坐标，每天做出一张图
	
	
	Arguments:
		date_start {int} -- 要统计的开始日期
		date_end {int} -- 要统计的结束日期
	
	Keyword Arguments:
		channels {list} -- 要统计的channel_id,以列表形式给出，-2 为统计总和 (default: {[-2]})
	"""
	# connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	# dates = utils.get_date_list(date_start,date_end)
	# sql = "select * from log_return_s_wja_1_percent where date = %s and channel_id = %s"
	# for channel in channels:
	# 	path = utils.get_figure_path("return_date_test", "channel_" + str(channel))
	# 	print(path)
	# 	for date in dates:
	# 		print("------------date:",date," channel:",channel,"--------------")
	# 		result = connection.query(sql,[date,channel])
	# 		if(len(result) != 0):
	# 			return_percent = []
	# 			for i in range(2,31):
	# 				return_percent.append(result[0][str(i) + 'day'])
	# 			# number = result[0][5:34]
	# 			days = range(2,31)
	# 			plt.plot(days,return_percent,'r--')
	# 			plt.gca().set_xlabel('days')
	# 			plt.gca().set_ylabel('return')
	# 			plt.grid(True)
	# 			#plt.savefig(os.path.join(path,str(date)))
	# 			plt.show()
	# 			plt.cla()

	# connection.close()


	days = range(2,31)
	dates = date_util.get_date_list(date_start,date_end)
	for channel in channels:
		for locale in locales:
			for date in dates:
				figure_path = file_util.get_figure_path("return_date_test", "channel_" + str(channel))
				log_tmp_path = file_util.get_log_tmp_path("return",game_id,date)
				log_file = os.path.join(log_tmp_path,"channel_" + str(channel) + "_locale_" + str(locale))
				print("------------date:",date," channel:",channel,"--------------")
				with open(log_file,'r') as f:
					return_percent = [float(x) for x in f.read().strip().split(" ")]
					plt.plot(days,return_percent,'r--')
					plt.gca().set_xlabel('days')
					plt.gca().set_ylabel('return')
					plt.grid(True)
					#plt.savefig(os.path.join(figure_path,str(date)))
					#plt.show()
					plt.cla()




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
		path = file_util.get_figure_path("return_day_test")
		plt.savefig(os.path.join(path,str(day) + "day"))
		#plt.show()
		plt.cla()

		fig.set_size_inches(10,5)
		plt.hist(return_percent)
		path = file_util.get_figure_path("return_day_test","hist")
		plt.savefig(os.path.join(path,str(day) + "day"))
		#plt.show()
	
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
	path = file_util.get_figure_path("dau and dnu test")
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
	path = file_util.get_figure_path("dau and dnu test")
	plt.savefig(os.path.join(path,"dau and dnu bar"),dpi=100)
	#plt.show()

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
	path = file_util.get_figure_path("dau and dnu")
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
		path = file_util.get_figure_path()
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
		#plt.show()
		plt.cla()

	connection.close()

if __name__ == '__main__':

	game_id = sys.argv[1]

	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	result_path = file_util.get_result_path("return", game_id)
	for file in os.listdir(result_path):
		full_file = os.path.join(result_path,file)
		if os.path.isfile(full_file):
			file_name_split = file.split('_')
			channel = file_name_split[1]
			locale = file_name_split[-1]
			with open(full_file,'r') as f:
				popt = f.read().split()

			result_table = db_util.get_result_table("return", game_id)
			sql = "delete from " + result_table + " where channel = %s and locale = %s"
			connection.query(sql,(channel, locale))
			sql = "insert into " + result_table + "(channel, locale, a, b, c) values (%s, %s, %s, %s, %s)"
			connection.query(sql,(channel, locale, popt[0], popt[1], popt[2]))

	connection.close()




