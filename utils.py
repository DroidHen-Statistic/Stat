import os
import config

def figure_path(*subfolder):
	path = os.path.join(config.base_dir,"figures")
	for folder in subfolder:
		path = os.path.join(path,folder)
		if not os.path.exists(path):
			 os.mkdir(path)
	return path

def item_user_table(game_id):
	return "user_item_s_" + str(game_id)

def item_item_table(game_id):
	return "item_item_s_" + str(game_id)

def log_dir(log_type, game_id):
	return config.log_base_dir + "\\"  + log_type +"\\s_" + str(game_id) + "\\" + log_type +"_2"

def union_dict(*objs):
	keys = set(sum([list(obj.keys()) for obj in objs],[]))  
	total = {}  
	for key in keys:  
		total[key] = sum([obj.get(key,0) for obj in objs])  
	return total  