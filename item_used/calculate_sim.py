import sys
sys.path.append("..")

import config
from MysqlConnection import MysqlConnection
import numpy as np
import utils

def centralized(l):
	"""
	中心化向量
	
	Arguments:
		l {list} -- 要中心化的向量
	
	Returns:
		list -- 中心化后的向量
	"""
	average = sum(l)/len(l)
	for i in range(len(l)):
		l[i] = l[i] - average
	return l

def e_length(l):
	"""
	向量模长
	
	Arguments:
		l {nparray} -- 输入向量
	
	Returns:
		number -- 向量模长
	"""
	return np.sqrt(l.dot(l))

def sim_cosin(i,j):
	"""
	余弦相似度
	
	Arguments:
		i {nparray} -- 向量i
		j {nparray} -- 向量j
	
	Returns:
		number -- i和j的余弦相似度
	"""
	return i.dot(j)/(e_length(i) * e_length(j))

def sim_person(i,j):
	"""
	Person相关性系数

	Arguments:
		i {nparray} -- 向量i
		j {nparray} -- 向量j
	
	Returns:
		number -- i和j的Person相关性系数
	"""
	i = centralized(i)
	j = centralized(j)
	return i.dot(j)/(e_length(i) * e_length(j))



def sim():
	"""
	计算item之间的相似度，这里采用的是余弦相似度
	
	"""
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassward,config.dbname)
	item_user_table = utils.item_user_table(game_id)
	item_item_table = utils.item_item_table(game_id)
	sql = "select column_name from Information_schema.columns where table_Name = %s"
	columns = connection.query(sql,item_user_table)
	columns = list(zip(*columns))[0][1:]
	sql = "select * from " + item_user_table
	result = connection.query(sql)
	R_u = [x[1:] for x in result]
	R_u_centralized = list(map(centralized,R_u))
	#items = np.array(list(zip(*R_u_centralized)))
	items = np.array(list(zip(*R_u)))
	#print(items)
	for i in range(len(items)):
		for j in range(i + 1,len(items)):
			sim_ij = sim_cosin(items[i],items[j])
			print("-------",i,j,"-------")
			sql = "update " + item_item_table + " set " + columns[j] + " = %s where item_id = %s"
			print(sql)
			print(sim_ij,columns[i])
			connection.query(sql,[float(sim_ij),columns[i][5:]])
			#print(sim_ij)

	connection.close()



if __name__ == '__main__':
	game_id = sys.argv[1]
	sim()

