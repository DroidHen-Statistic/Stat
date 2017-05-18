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
	return_num = {}
	data = []
	for date in dates:
		result = connection.query(sql,date)
		return_num[date] = result[0][5:34]
		data = data + result[0][5:34]
	plt.plot(days * len(dates), data,'o')

	popt, pcov = curve_fit(func, days * len(dates), data)
	y = [func(day,popt[0],popt[1],popt[2]) for day in days]
	plt.plot(days, y,'b')
	plt.show()
	plt.cla()

	for date in dates:
		print("------",date,"------")
		plt.plot(days,return_num[date],'r--',label = 'origin')
		plt.plot(days,y,'b--',label = 'fit curve')
		plt.legend(loc = 'upper right')
		plt.grid(True)
		#plt.show()
		#plt.savefig("E:/python/stat/figures/return_date_fit/all/"+str(date)+".jpg")
		plt.cla()

	connection.close()
	return popt, pcov

if __name__ == '__main__':
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	sql = "select date from log_return_s_wja_1_percent"
	result = connection.query(sql)
	dates = sorted(list(set(reduce(lambda x,y : x + y, result))))
	connection.close()

	popt,pcov = fitDateReturn(dates[5:-29])
	print(popt,pcov)
