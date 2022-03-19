import modules.mysql_connection

def sqlQuery(sql):
    (cursor, cnx) = modules.mysql_connection.get_cursor()
    cursor.execute(sql)
    t_data=cursor.fetchall()
    return t_data

def sqlQueryComment(button_name, comment_level):
    (cursor, cnx) = modules.mysql_connection.get_cursor()
    sql = (f"select A.comment from comment_table A join button_table B on A.button_id = B.button_id where A.comment_level = {comment_level} and B.button_name = '{button_name}';")
    cursor.execute(sql)
    t_data=cursor.fetchall()
    return t_data

# def sqlQueryTest(param, table_name):
#     (cursor, cnx) = modules.mysql_connection.get_cursor()
#     # param = 1
#     sql = (f"SELECT face_name FROM {table_name} where face_id = {param}")
#     cursor.execute(sql)
#     t_data=cursor.fetchall()
#     return t_data
