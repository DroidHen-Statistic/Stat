import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils
from MysqlConnection import MysqlConnection
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy import log
from functools import reduce
import os

from sklearn.model_selection import KFold

def func(x,a,b,c):
	return a * (b ** x) + c

def average(l):
	return sum(l)/len(l)

def fitDateReturn(game_id, channels = [-2]):
	"""
	拟合每天新用户的30日留存曲线

	采用scipy 的 curve_fit，原理是最小二乘

	利用10折交叉验证，选取了error最小的一次作为最终模型
	
	Arguments:
		game_id {int} -- 游戏id
	
	Keyword Arguments:
		channels {list} -- 要拟合的渠道信息，-2为总和 (default: {[-2]})
	
	Returns:
		best_popt {list} -- 拟合得到的参数值
		best_pcov {list} -- 系数的协方差值，后续暂时没有用到
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)
	days = list(range(2,31))
	for channel in channels:
		sql = "select date from log_return_s_wja_1_percent where channel_id = %s"
		result = connection.query(sql,channel)
		dates = [x['date'] for x in result]
		popts = []
		return_num = {}
		all_data = np.ones(29);
		path = utils.get_figure_path("return_date_fit","10-fold","channel_" + str(channel))
		sql = "select * from log_return_s_wja_1_percent where date = %s and channel_id = %s"
		for date in dates:
			result = connection.query(sql,[date,channel])
			if(len(result)!=0):
				return_percent= []
				for i in range(2,31):
					return_percent.append(result[0][str(i) + 'day'])
				# return_num[date] = return_percent
				if return_percent.count(0) < 5:
					all_data = np.row_stack((all_data,return_percent))
				else:
					dates.remove(date)
			else:
				dates.remove(date)

		# y是一个矩阵，每一行为某一日的2到30日留存值，x与y的维数相等，每一行都是2,3,...,29
		y = all_data[1:] 	#这里因为all_data第一行不是真实数据，而是全部为1
		x = np.array(days)
		x = np.tile(x,(y.shape[0],1))

		popts = []
		min_err = float("inf")
		min_i = 0

		if(y.shape[0] < 10):
			print("The data size is too small(less than 10)")
			break;
		'''
		10折交叉验证，这里直接使用了sklearn中的KFold接口，用于将数据分为10份
		每次训练使用其中的9份作为训练数据，其中的1份作为验证集来对模型进行评价，最后选取效果最好的一个
		'''
		kf = KFold(n_splits=10)
		for train, test in kf.split(y):		#这里train 和 test分别保存了训练集和验证集在数据集中的下标，所以可以直接里利用该下标来取出对应的数据
			x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
			popt, pcov = curve_fit(func, x_train.ravel(), y_train.ravel())	#ravel()函数将原本的二维数据展开为一维
			err_sum = 0
			y_hat = np.array([func(day,popt[0],popt[1],popt[2]) for day in days])
			for i in range(y_test.shape[0]):
				err_sum += sum((y_test[i] - y_hat)**2)
			if err_sum < min_err:
				min_err = err_sum
				best_popt = popt
				best_pcov = pcov


	
		'''10折交叉验证'''
		# size = int(len(data)/DAYS/10)
		# samples = int(len(data)/DAYS)
		# for i in range(0,10):
		# 	if i == 0:
		# 		data_train = data[size * DAYS:]
		# 		data_test = data[0:size * DAYS]
		# 	else:
		# 		data_train = data[0 : i * size * DAYS] + data[ (i + 1) * size * DAYS :]
		# 		data_test = data[i * size * DAYS : (i + 1) * size * DAYS]
		# 	#plt.plot(days * (len(dates) - size), data_train,'o')
		# 	popt, pcov = curve_fit(func, days * (samples-size), data_train)
		# 	print(len(data_test))
		# 	err_sum = 0
		# 	y_hat = [func(day,popt[0],popt[1],popt[2]) for day in days]
		# 	for j in range(0,9):#TODO: 这里是个bug，不应该是（0，9）
		# 		y = data_test[j * DAYS : (j+1) * DAYS]
		# 		err = list(map(lambda x:pow(x[0] - x[1],2), zip(y,y_hat)))
		# 		err_sum = err_sum + reduce(lambda x,y : x + y,err)
		# 	if err_sum < min_err:
		# 		min_err = err_sum
		# 		min_i = i
		# 	popts.append(popt)

		# popt = popts[min_i]

		'''计算拟合曲线'''
		y_hat = [func(day,best_popt[0],best_popt[1],best_popt[2]) for day in days]
		plt.plot(days, y_hat,'b')
		plt.show()
		plt.cla()

		'''画出每天的图像以及拟合曲线的图像'''
		# for date ,num in return_num.items():
		# 	print("------",date,"------")
		# 	plt.plot(days,num,'r--',label = 'origin')
		# 	plt.plot(days,y,'b--',label = 'fit curve')
		# 	plt.legend(loc = 'upper right')
		# 	plt.grid(True)
		# 	#plt.show()
		# 	plt.savefig(os.path.join(path,str(date)))
		# 	plt.cla()
		# 	
		for i in range(len(dates)):
			print("------",dates[i],"------")
			plt.plot(days,y[i],'r--',label = 'origin')
			plt.plot(days,y_hat,'b--',label = 'fit curve')
			plt.legend(loc = 'upper right')
			plt.grid(True)
			plt.show()
			#plt.savefig(os.path.join(path,str(date)))
			plt.cla()


	connection.close()
	return best_popt, best_pcov

if __name__ == '__main__':

	DAYS = 29
	game_id = sys.argv[1]

	popt,pcov = fitDateReturn(game_id)

	print(popt,pcov)