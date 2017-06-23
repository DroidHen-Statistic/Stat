import pymysql
import pymysql.cursors

class MysqlConnection(object):
    def __init__(self,host,user,password,db,charset = 'utf8',cursorclass = pymysql.cursors.DictCursor):
        self.__connection = pymysql.connect(host=host,user=user,password=password,db=db,charset=charset,cursorclass = cursorclass)

    def close(self):
        self.__connection.close()

    def query(self,sql,values=()):
        result = []
        #获取一个游标
        with self.__connection.cursor() as cursor:
            cout=cursor.execute(sql,values)
            self.__connection.commit()
            for row in cursor.fetchall():
                #result.append(list(row))
                result.append(row)
                #print('%s\t%s\t%s' %row
        return result

