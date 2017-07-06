# def get_item_user_table(game_id):
# 	return "user_item_" + game_id

def get_item_item_table(game_id):
	return "output_item_item_" + game_id

def get_log_table(log_type, game_id, server_id = -1):
	return "log_" + log_type + "_s_wja_" + game_id +"_" + str(server_id)

def get_result_table(log_type, game_id):
	return "output_" + log_type + "_" + str(game_id)

