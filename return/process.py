import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils import *
from MysqlConnection import MysqlConnection
from scipy.optimize import curve_fit
import numpy as np
from scipy import log
from functools import reduce
import os
import platform

from sklearn.model_selection import KFold

def func(x,a,b,c):
	return a * (b ** x) + c

# def func(x,k,l,p):
# 	return p * (k/l)*((x/l) ** (k-1)) * np.exp(-(x/l)**k)

def average(l):
	return sum(l)/len(l)

def fitDateReturn(game_id, locales = [-2], channels = [-2]):
	"""
	拟合每天新用户的30日留存曲线

	采用scipy 的 curve_fit，原理是最小二乘

	利用10折交叉验证，选取了error最小的一次作为最终模型

	最后的结果保存到文件中
	
	Arguments:
		game_id {int} -- 游戏id
	
	Keyword Arguments:
		channels {list} -- 要拟合的渠道信息，-2为总和 (default: {[-2]})
		locales {list} -- 要拟合的渠道信息，-2为总和 (default: {[-2]})
	"""
	# connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	# days = list(range(2,31))
	# log_type_tmp_path = utils.get_log_type_tmp_path("return",game_id)
	# dates = os.listdir(log_type_tmp_path)
	# for channel in channels:
	# 	for locale in locales:
	# 		all_data = np.ones(29)
	# 		sql = "select * from log_return_s_wja_1_percent where channel_id = %s and locale = %s"
	# 		result = connection.query(sql,[channel,locale])
	# 		dates = []
	# 		for i in range(len(result)):
	# 			return_percent= []
	# 			for j in range(2,31):
	# 				return_percent.append(result[i][str(j) + 'day'])
	# 			if return_percent.count(0) < 5:
	# 				all_data = np.row_stack((all_data,return_percent))
	# 				dates.append(result[i]["date"])
	# 			else:
	# 				print("data dropped: %s %s %s" %(channel,locale,result[i]['date']))

			

	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	days = list(range(2,31))
	log_type_tmp_path = file_util.get_log_type_tmp_path("return",game_id)
	date_list = os.listdir(log_type_tmp_path)
	for channel in channels:
		for locale in locales:
			all_data = np.ones(29)
			dates = []
			for date in date_list:
				log_tmp_path = os.path.join(log_type_tmp_path,date)
				log_file = os.path.join(log_tmp_path, "channel_" + str(channel) + "_locale_" + str(locale))
				with open(log_file, 'r') as f:
					return_percent = [float(x) for x in f.read().strip().split(" ")]
					if return_percent.count(0) < 5:
						all_data = np.row_stack((all_data,return_percent))
						dates.append(date)
					else:
						print("data dropped: %s %s %s" %(channel,locale,date))


			path = file_util.get_figure_path("return_date_fit","channel_" + str(channel) + "_locale_" + str(locale))

			if(all_data.shape[0] < 11 or len(all_data.shape) == 1):
				print("The valid data size is too small (less than 10) for channel %d locale %s" %(channel,locale))
				break

			# y是一个矩阵，每一行为某一日的2到30日留存值，x与y的维数相等，每一行都是2,3,...,29
			y = all_data[1:] 	#这里因为all_data第一行不是真实数据，而是全部为1
			x = np.array(days)
			x = np.tile(x,(y.shape[0],1))

			min_err = float("inf")
			min_i = 0

			'''
			10折交叉验证，这里直接使用了sklearn中的KFold接口，用于将数据分为10份
			每次训练使用其中的9份作为训练数据，其中的1份作为验证集来对模型进行评价，最后选取效果最好的一个
			'''
			kf = KFold(n_splits=10)
			for train, test in kf.split(y):		#这里train 和 test分别保存了训练集和验证集在数据集中的下标，所以可以直接里利用该下标来取出对应的数据
				x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
				popt, pcov = curve_fit(func, x_train.ravel(), y_train.ravel(), bounds = ((0,0,0),(np.inf,1,np.inf)))	#ravel()函数将原本的二维数据展开为一维
				err_sum = 0
				y_hat = np.array([func(day,popt[0],popt[1],popt[2]) for day in days])
				# y_hat = np.array([func(day,popt[0],popt[1]) for day in days])
				for i in range(y_test.shape[0]):
					err_sum += sum((y_test[i] - y_hat) ** 2)
				if err_sum < min_err:
					min_err = err_sum
					best_popt = popt
					best_pcov = pcov

			'''计算拟合曲线'''
			y_hat = [func(day,best_popt[0],best_popt[1],best_popt[2]) for day in days]
			# y_hat = [func(day,best_popt[0],best_popt[1]) for day in days]
			plt.plot(days, y_hat,'b')
			# plt.show()
			plt.cla()

			for i in range(len(dates)):
				print("------",dates[i],"------")
				plt.plot(days,y[i],'r--',label = 'origin')
				plt.plot(days,y_hat,'b--',label = 'fit curve')
				plt.legend(loc = 'upper right')
				plt.gca().set_xlabel("days")
				plt.gca().set_ylabel("return_percent")
				plt.grid(True)
				# plt.show()
				plt.savefig(os.path.join(path,str(dates[i])))
				plt.cla()

			print("result for channel %d locale %s is %f * (%f ** x) + %f" %(channel, locale, best_popt[0],best_popt[1],best_popt[2]))
			result_path = file_util.get_result_path("return", game_id)
			result_file = os.path.join(result_path,"channel_" + str(channel) + "_locale_" + str(locale))
			with open(result_file, 'w') as f:
				for p in best_popt:
					f.write(str(p) + " ")
				# f.write("%f * (%f ** x) + %f" %(best_popt[0],best_popt[1],best_popt[2]))

	connection.close()
		

if __name__ == '__main__':

	if platform.system() == "Linux":
		import matplotlib
		matplotlib.use('agg')
	import matplotlib.pyplot as plt
	DAYS = 29
	# game_id = sys.argv[1]
	game_id = "s_101250"

	fitDateReturn(game_id)
	# x = np.arange(20)
	# y = func(x,0.5,1)
	# plt.plot(x,y)
	# plt.show()
