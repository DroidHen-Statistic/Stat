from MysqlConnection import MysqlConnection

conn = MysqlConnection("218.108.40.13","wja","wja","wja")
sql = "select * from test where username = %s"
result = conn.query(sql,"wja")
print(result)



