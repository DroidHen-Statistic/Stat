def item_user_table(game_id):
	return "user_item_" + game_id

def item_item_table(game_id):
	return "item_item_" + game_id

def get_log_table(log_type, game_id, server_id = -1):
	return "log_" + log_type + "_s_wja_" + game_id +"_" + str(server_id)
