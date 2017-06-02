import os

def item_user_table(game_id):
	return "user_item_s_" + str(game_id)

def log_dir(log_type, game_id):
	return "E:/log/"+ log_type +"/s_" + str(game_id) + "/" + log_type +"_2"