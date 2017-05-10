from MysqlConnection import MysqlConnection
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from functools import reduce

def func(x,a,b):
	return a*(b**x)

def fitDateReturn(dates):
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	sql = "select * from log_return_s_wja_1_percent where date = %s"
	for date in dates:
		result = connection.query(sql,date)
		return_num = result[0][5:34]
		for i in range(len(return_num)):
			return_num[i] /= 100
		days = list(range(2,31))
		plt.plot(days,return_num,'r--')
		popt, pcov = curve_fit(func, days[0:7], return_num[0:7])
		print(popt)
		y = [func(day,popt[0],popt[1]) for day in days]
		plt.plot(days,y,'b--')
		plt.grid(True)
		#plt.show()
		plt.savefig("E:/python/stat/figures/return_date_fit/"+str(date)+".jpg")
		plt.cla()
	connection.close()
	return popt

if __name__ == '__main__':
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	sql = "select date from log_return_s_wja_1_percent"
	result = connection.query(sql)
	dates = sorted(list(set(reduce(lambda x,y : x + y, result))))
	connection.close()

	popt = fitDateReturn(dates)
