BEGIN{
#	print total_id
}
{
    uid = $1
    item = $3
    count = $4
    uid_item_count[uid,item]++
}
END{
	# 当天的uid列表
	for( key in uid_item_count)
	{
		split (key,info,SUBSEP)
		{
			uid=info[1];
            if(!uid_info[uid])
            {
                uid_info[uid] = uid 
            }
			item=info[2];
			count=uid_item_count[key];
            uid_info[uid] = uid_info[uid]" "item" "count
#            print uid
#            print uid" "item" "count > out_file
		}
	}
    delete uid_item_count
    for( uid in uid_info)
    {
        print uid_info[uid] > out_file
    }
}
