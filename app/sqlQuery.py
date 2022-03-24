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

def sqlQueryHairMakeupComment(button_name, hair_makeup_level):
    (cursor, cnx) = modules.mysql_connection.get_cursor()
    sql = (f"select A.hair_makeup_comment from hair_makeup_table A join button_table B on A.button_id = B.button_id where A.hair_makeup_level = {hair_makeup_level} and B.button_name = '{button_name}';")
    cursor.execute(sql)
    t_data=cursor.fetchall()
    return t_data

def sqlQueryMember(user_id = '', password = ''):
    (cursor, cnx) = modules.mysql_connection.get_cursor()
    if len(password) != 0:
        sql = (f"select * from members_table where user_id = '{user_id}' and password = '{password}';")
    elif len(user_id) != 0:
        sql = (f"select * from members_table where user_id = '{user_id}';")
    else:
        return False
    cursor.execute(sql)
    t_data=cursor.fetchall()
    if len(t_data) != 0:
        return True
    else:
        return False

def sqlInsertMember(user_id = '', password = ''):
    (cursor, cnx) = modules.mysql_connection.get_cursor()
    # if len(password) != 0:
    #     sql = (f"select * from members_table where user_id = '{user_id}' and password = '{password}';")
    # elif len(user_id) != 0:
    #     sql = (f"select * from members_table where user_id = '{user_id}';")
    # else:
    #     return False
    sql = (f"insert into members_table (user_id, password) values ('{user_id}', '{password}');")
    print('-------')
    print(sql)
    print('-------------')
    cursor.execute(sql)
    t_data=cursor.fetchall()
    print(t_data)
    print('-------')
    if len(t_data) != 0:
        return True
    else:
        return False

# def sqlQueryTest(param, table_name):
#     (cursor, cnx) = modules.mysql_connection.get_cursor()
#     # param = 1
#     sql = (f"SELECT face_name FROM {table_name} where face_id = {param}")
#     cursor.execute(sql)
#     t_data=cursor.fetchall()
#     return t_data
