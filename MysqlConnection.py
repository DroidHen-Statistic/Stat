import pymysql
import pymysql.cursors

class MysqlConnection(object):
    def __init__(self,host,user,password,db,charset = 'utf8'):
        self.__connection = pymysql.connect(host=host,user=user,password=password,db=db,charset=charset)

    def close(self):
        self.__connection.close()

    def query(self,sql,values=()):
        result = []
        #获取一个游标
        with self.__connection.cursor() as cursor:
            cout=cursor.execute(sql,values)
            #cursor.execute("select * from log_level_left_s_wja_1 where date = %s", (20161221))
            self.__connection.commit()
            for row in cursor.fetchall():
                result.append(list(row))
                #print('%s\t%s\t%s' %row
                #注意int类型需要使用str函数转义
        return result