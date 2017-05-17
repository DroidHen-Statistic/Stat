from MysqlConnection import MysqlConnection
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import math
from scipy import log
from functools import reduce

def func(x,a,b,c):
	return a * (b ** x) + c

def average(l):
	return sum(l)/len(l)

def fitDateReturn(dates):
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	sql = "select * from log_return_s_wja_1_percent where date = %s"
	popts = []
	days = list(range(2,31))
	return_num = []
	for date in dates:
		result = connection.query(sql,date)
		return_num = result[0][5:34]
		# for i in range(len(return_num)):
		# 	return_num[i] /= 100
		plt.plot(days,return_num,'r--')
		popt, pcov = curve_fit(func, days, return_num)
		popts.append(popt)
		print("------",date,"------")
		print(popt)
		y = [func(day,popt[0],popt[1],popt[2]) for day in days]
		#y = [func(day,popt[0]) for day in days]
		plt.plot(days,y,'b--')
		plt.grid(True)
		#plt.show()
		plt.savefig("E:/python/stat/figures/return_date_fit/all/"+str(date)+".jpg")
		plt.cla()

	popts = list(zip(*popts))
	popts = list(map(average,popts))
	y = [func(day,popts[0],popts[1],popts[2]) for day in days]

	for date in dates:
		result = connection.query(sql,date)
		return_num = result[0][5:34]
		# for i in range(len(return_num)):
		# 	return_num[i] /= 100
		plt.plot(days,return_num,'r--')
		plt.plot(days,y,'b--')
		plt.grid(True)
		plt.savefig("E:/python/stat/figures/return_date_fit/average/"+str(date)+".jpg")
		plt.cla()

	connection.close()
	return popt, popts

if __name__ == '__main__':
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	sql = "select date from log_return_s_wja_1_percent"
	result = connection.query(sql)
	dates = sorted(list(set(reduce(lambda x,y : x + y, result))))
	connection.close()

	[popt,popts] = fitDateReturn(dates[5:-29])
