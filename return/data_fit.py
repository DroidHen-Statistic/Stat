import sys
sys.path.append("..")

from MysqlConnection import MysqlConnection
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import math
from scipy import log
from functools import reduce
import os

DAYS = 29

def func(x,a,b,c):
	return a * (b ** x) + c

def average(l):
	return sum(l)/len(l)

def fitDateReturn(dates,channels = [-2]):
	"""
	拟合每天新用户的30日留存曲线

	采用scipy 的 curve_fit，原理是最小二乘

	利用10折交叉验证，选取了error最小的一次作为最终模型
	
	Arguments:
		dates {list} -- 数据样本的日期
	
	Keyword Arguments:
		channels {list} -- 要拟合的channel_id列表，-2为所有channel总和 (default: {[-2]})
	
	Returns:
		list -- 模型的拟合参数
	"""
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	sql = "select * from log_return_s_wja_1_percent where date = %s and channel_id = %s"
	days = list(range(2,31))
	for channel in channels:
		popts = []
		return_num = {}
		data = []
		path = utils.figure_path("return_date_fit","10-fold","channel_" + str(channel))
		for date in dates:
			result = connection.query(sql,[date,channel])
			if(len(result)!=0):
				return_num[date] = result[0][5:34]
				data = data + result[0][5:34]

		popts = []
		min_err = float("inf")
		min_i = 0
	
		'''10折交叉验证'''
		size = int(len(data)/DAYS/10)
		samples = int(len(data)/DAYS)
		for i in range(0,10):
			if i == 0:
				data_train = data[size * DAYS:]
				data_test = data[0:size * DAYS]
			else:
				data_train = data[0 : i * size * DAYS] + data[ (i + 1) * size * DAYS :]
				data_test = data[i * size * DAYS : (i + 1) * size * DAYS]
			#plt.plot(days * (len(dates) - size), data_train,'o')
			popt, pcov = curve_fit(func, days * (samples-size), data_train)	
			print(len(data_test))
			err_sum = 0
			y_hat = [func(day,popt[0],popt[1],popt[2]) for day in days]
			for j in range(0,9):
				y = data_test[j * DAYS : (j+1) * DAYS]
				err = list(map(lambda x:pow(x[0] - x[1],2), zip(y,y_hat)))
				err_sum = err_sum + reduce(lambda x,y : x + y,err)
			if err_sum < min_err:
				min_err = err_sum
				min_i = i
			popts.append(popt)


		#popts = list(zip(*popts))
		#popt = list(map(average,popts))
		popt = popts[min_i]

		'''计算拟合曲线'''
		y = [func(day,popt[0],popt[1],popt[2]) for day in days]
		plt.plot(days, y,'b')
		plt.show()
		plt.cla()

		'''画出每天的图像以及拟合曲线的图像'''
		for date ,num in return_num.items():
			print("------",date,"------")
			plt.plot(days,num,'r--',label = 'origin')
			plt.plot(days,y,'b--',label = 'fit curve')
			plt.legend(loc = 'upper right')
			plt.grid(True)
			#plt.show()
			plt.savefig(os.path.join(path,str(date)))
			plt.cla()

	connection.close()
	return popt, pcov

if __name__ == '__main__':
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	sql = "select date from log_return_s_wja_1_percent"
	result = connection.query(sql)
	dates = sorted(list(set(reduce(lambda x,y : x + y, result))))
	connection.close()

	popt,pcov = fitDateReturn(dates[6:-29],[-1,3,66])

	print(popt,pcov)