import os
import config

from datetime import datetime, timedelta
from functools import reduce


def date_to_int(date):
	return int(datetime.strftime(date,"%Y%m%d"))

def int_to_date(date):
	return datetime.strptime(str(date),"%Y%m%d")

def get_yesterday(date):
	return date_to_int(int_to_date(date) - timedelta(days = 1))

def get_date_after_n(date, n):
    return date_to_int(int_to_date(date) + timedelta(days = n))

def gen_dates(b_date, days):
    day = timedelta(days=1)
    for i in range(days):
        yield b_date + day * i

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

def split_date(date):
	date_str = str(date)
	year = date_str[0:4]
	month = date_str[4:6]
	day = date_str[6:8]
	return year,month,day