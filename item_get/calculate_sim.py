import sys
sys.path.append("..")

from MysqlConnection import MysqlConnection
import numpy as np

def centralized(l):
	average = sum(l)/len(l)
	for i in range(len(l)):
		l[i] = l[i] - average
	return l

def e_length(l):
	return np.sqrt(l.dot(l))

def sim_cosin(i,j):
	return i.dot(j)/(e_length(i) * e_length(j))

def sim_person(i,j):
	i = centralized(i)
	j = centralized(j)
	return i.dot(j)/(e_length(i) * e_length(j))



def sim():
	connection = MysqlConnection("218.108.40.13","wja","wja","wja")
	sql = "select * from user_item"
	result = connection.query(sql)
	R_u = [x[1:] for x in result]
	R_u_centralized = list(map(centralized,R_u))
	#items = np.array(list(zip(*R_u_centralized)))
	items = np.array(list(zip(*R_u)))
	for i in range(len(items)):
		for j in range(i + 1,len(items)):
			sim_ij = sim_cosin(items[i],items[j])
			print(sim_ij)

	connection.close()



if __name__ == '__main__':
	sim()

