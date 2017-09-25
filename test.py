from utils import other_util

file = "ipdb.csv"
ipdb = other_util.IPDB(file)
ret = ipdb.ip2cc("41.202.207.9")
print(ret)