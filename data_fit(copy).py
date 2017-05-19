from MysqlConnection import MysqlConnection
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import math
from scipy import log
from functools import reduce

DAYS = 29

def func(x,a,b,c):
	return a * (b ** x) + c

def average(l):
	return sum(l)/len(l)

def fitDateReturn(dates):
	"""
	拟合每日新增用户的30日留存曲线
	采用scipy 的 curve_fit，原理是最小二乘
	利用10折交叉验证，选取了error最小的一次作为最终模型
	拟合曲线以及原始数据都保存在当前目录下的figure/return-date-fit/10-fold文件夹下
	"""
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	sql = "select * from log_return_s_wja_1_percent where date = %s"
	popts = []
	days = list(range(2,31))
	return_num = {}
	data = []
	for date in dates:
		result = connection.query(sql,date)
		return_num[date] = result[0][5:34]
		data = data + result[0][5:34]

	popts = []
	min_err = float("inf")
	min_i = 0
	
	'''10折交叉验证'''
	for i in range(0,int(len(dates)/10)):
		if i == 0:
			data_train = data[10 * DAYS:]
			data_test = data[0:10 * DAYS]
		else:
			data_train = data[0 : i * 10 * DAYS] + data[ (i + 1) * 10 * DAYS :]
			data_test = data[i * 10 * DAYS : (i + 1) * 10 * DAYS]
		#plt.plot(days * (len(dates) - 10), data_train,'o')
		popt, pcov = curve_fit(func, days * (len(dates)-10), data_train)	
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
	for date in dates:
		print("------",date,"------")
		plt.plot(days,return_num[date],'r--',label = 'origin')
		plt.plot(days,y,'b--',label = 'fit curve')
		plt.legend(loc = 'upper right')
		plt.grid(True)
		#plt.show()
		plt.savefig(os.path.dirname(__file__) +  "/figures/return_date_fit/10-fold/"+str(date)+".jpg")
		plt.cla()

	connection.close()
	return popt, pcov

if __name__ == '__main__':
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	sql = "select date from log_return_s_wja_1_percent"
	result = connection.query(sql)
	dates = sorted(list(set(reduce(lambda x,y : x + y, result))))
	connection.close()

	popt,pcov = fitDateReturn(dates[6:-29])
	print(popt,pcov)