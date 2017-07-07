import sys
import os
from sys import argv
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
#print(abs_path)
sys.path.append(abs_path)
#sys.path.append('../')
from utils import *
import numpy as np

#k = utils.item_user_table(20178912)
#print(k)

#def calc_relation(uid_item_cout, items):
#    uids = uid_item_cout.keys()
#    
#    print(uids)   
#    for uids in 
 
argc = len(argv)
#print(argc)
if ( argc < 4 ):
	print ("params: date_start(YYYYMMDD), date_end(include), game_id, [server_id = 1]")
	exit()
# file_name = argv[0]
date_start = argv[1]
date_end = argv[2]
game_id = argv[3]
server_id = -1
if ( argc > 4 ):
	server_id = argv[4]
log_type = 'item_used'
tmp_dir = file_util.get_log_tmp_path(log_type, game_id, date_start, server_id)
after_read = os.path.join(tmp_dir, "after_read")
uid_item_count={}
item_uid_count={}
with open(after_read, 'r') as f:
    items = {}
#    uids = []
    i_count = 0;
    for line in f.readlines():
        line = line.split()
#        print(line)
        uid = line[0]            
        uid_item_count[uid] = []
        line_len = len(line)
        for i in range(1, line_len - 1, 2):
            item_id = line[i]
            if item_id not in items:
                items[item_id] = 1
                item_uid_count[item_id]=[]
            count = int(line[i+1])
            item_uid_count[item_id].append((uid,count))
            uid_item_count[uid].append( {line[i]: line[i+1]})
        i_count += 1
#        print(i_count)
        if (i_count > 5):
            break
    print (item_uid_count)

    uids = list(uid_item_count.keys())
    print(uids)
    item_len = len(item_uid_count)
    uid_len = len(uid_item_count)
    matrix = np.zeros((item_len, uid_len))
    item_pos = 0
    for item in item_uid_count:
        info = item_uid_count[item]
#        print (info)
#        print (item)
        #v = zip(uid_count)
        for uid_count in info:
            cr_uid = uid_count[0]
            count = uid_count[1]
            uid_pos = uids.index(cr_uid)
            print("uid: %s count: %s uid_pos: %s item_pos: %s" % (cr_uid, count, uid_pos, item_pos ))
            matrix[item_pos][uid_pos] += count    
#            exit()
        item_pos += 1
    print (matrix)
            
    relation = np.zeros((item_len, item_len)) 
    for i in range(item_len):
        for j in range(i+1, uid_len - 1):
            data1 = matrix[i]
            data2 = matrix[j]
            corated = np.vstack((data1, data2))
#            corated = []
#            for k,v in enumerate(data1):
#                v2 = data2[k]
#                if( (v != 0) or (v2 != 0)):
#                    corated.append((v,v2))
#            corated = list(zip(data1, data2))
            print(corated)
            cov = np.corrcoef(corated, rowvar= True)
#            opav = np.xxx(data1, data2)
            relation[i][j]= cov[0,1]
            exit()
            
#    print (relation)        

        
#print(uid_item_count)
#calc_relation(uid_item_count)
#exit()
