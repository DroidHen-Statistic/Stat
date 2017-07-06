import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from MysqlConnection import MysqlConnection
import numpy as np
from utils import *
import copy
import pickle

def centralized(l):
	"""
	中心化向量
	
	Arguments:
		l {list} -- 要中心化的向量
	
	Returns:
		list -- 中心化后的向量
	"""
	average = float(sum(l)/len(l))
	#print(average)
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
	# i = centralized(i)
	# j = centralized(j)
	
	if i_len == 0 or j_len ==0:
		return 0
	else:
		return i.dot(j)/(e_length(i) * e_length(j))



def sim(game_id):
	"""
	计算item之间的相似度，这里采用的是pearson相关性系数
	
	Arguments:
		game_id {int} -- game_id
	"""

	
	connection = MysqlConnection(config.dbhost,config.dbuser,config.dbpassword,config.dbname)

	# sql = "select column_name from Information_schema.columns where table_Name = %s"
	# columns = connection.query(sql,item_user_table)
	# columns = list(zip(*columns))[0][1:]
	

	

	
	# 文件接口
	# item_item_table = db_util.item_item_table(game_id)
	log_type_tmp_path = file_util.get_log_type_tmp_path("item_used", game_id)
	max_date = max(os.listdir(log_type_tmp_path))
	total_file = file_util.item_used_total_file(game_id, max_date)
	with open(total_file,'rb') as f:
		total_data = pickle.load(f)

	items = list(set(sum([list(x.keys()) for x in total_data.values()],[])))	#拿到所有的item id
	items = sorted(items, key = lambda x: int(x[5:]))	#这里为了最后的结果矩阵是个上三角阵，所以排了一下序
	item_item_relation = {}
	for i in range(len(items)):
		item_item_relation[items[i]] = {}
		for j in range(i,len(items)):
			print("------%s:%s------" %(items[i],items[j]))
			corated = []
			for k,v in total_data.items():
				if items[i] in v or items[j] in v:
					corated.append([v.get(items[i],0), v.get(items[j],0)])
			corated = np.array(corated)
			print(len(corated))
			if len(corated) != 0:
				corr_coef = np.corrcoef(corated, rowvar = False) 	#相关系数
				# sim_ij = corr_coef[0,1] if corr_coef[0,1] != np.nan else 0
				item_item_relation[items[i]][items[j]] = corr_coef[0,1] if corr_coef[0,1] != np.nan else 0
			else:
				# sim_ij = 0
				item_item_relation[items[i]][items[j]] = 0
			# sql = "update " + item_item_table + " set " + items[j] + " = %s where item_id = %s"
			print(item_item_relation[items[i]][items[j]])
			# connection.query(sql,[float(sim_ij),items[i][5:]])
			#print(sim_ij)
	
	result_path = file_util.get_result_path("item_used",game_id)
	with open(os.path.join(result_path, "result"),'wb') as f:
		pickle.dump(item_item_relation, f)





	# 数据库接口
	# item_user_table = db_util.item_user_table(game_id)
	# 
	# sql = "select * from " + item_user_table
	# result = connection.query(sql)
	# columns = list(result[0].keys())[1:]
	# user_item_table = np.array([list(x.values())[1:] for x in result])
	# cols = user_item_table.shape[1]
	# for i in range(cols):
	# 	for j in range(i, cols):
	# 		print("-------",i,j,"-------")
	# 		corated = []
	# 		for k in range(len(user_item_table[:,i])):
	# 			if user_item_table[k,i] != 0 or user_item_table[k,j] != 0:
	# 				corated.append([user_item_table[k,i],user_item_table[k,j]])
	# 			# if user_item_table[k,i] != 0 and user_item_table[k,j] != 0:
	# 			# 	corated.append([user_item_table[k,i],user_item_table[k,j]])
	# 		corated = np.array(corated)
	# 		print(len(corated))
	# 		if len(corated) != 0:
	# 			corr_coef = np.corrcoef(corated, rowvar = False) 	#协方差矩阵
	# 			print(corr_coef)
	# 			# sim_ij = corr_coef[0,1]/np.sqrt(corr_coef[0,0] * corr_coef[1,1])	#相关系数
	# 			sim_ij = corr_coef[0,1] if corr_coef[0,1] != np.nan else 0
	# 		else:
	# 			sim_ij = 0
	# 		#sim_ij = sim_cosin(user_item_table[i],user_item_table[j])
	# 		sql = "update " + item_item_table + " set " + columns[j] + " = %s where item_id = %s"
	# 		print(sql)
	# 		print(sim_ij,columns[i])
	# 		connection.query(sql,[float(sim_ij),columns[i][5:]])
	# 		#print(sim_ij)



	connection.close()



if __name__ == '__main__':
	
	game_id = sys.argv[1]
	sim(game_id)