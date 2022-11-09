import pandas as pd
import os
import pymysql
import numpy as np

def db_ssh_tunneling():
    path_ssh = 'local ssh 위치'
    os.chdir(path_ssh)
    os.system('ssh.exe')
    path_pem = 'ssh key file'
    cmd = 'ssh "%s"'%path_pem
    os.system('start cmd /k' + cmd)


def db_connecting():
    global db
    global cursor
    db = pymysql.connect(
        host='127.0.0.1',
        port = 'port_name',
        user='user',
        password='password',
        database='database_table',
        charset='utf8'
        )
    cursor = db.cursor(pymysql.cursors.DictCursor)


def insert_into_sql(table_name, df):
    
    sql = 'delete from ' + table_name
    sql = sql + ' where condition1'
    sql = sql + ' and condition2 ' + str(df.created_at[0])
    cursor.execute(sql)
    db.commit()
    
    val = df.to_records(index=False).tolist()
    sql = 'INSERT IGNORE INTO ' + table_name + ' VALUES (%s, %s, %s, %s, %s)'
    cursor.executemany(sql, val)
    db.commit()

db_ssh_tunneling()
db_connecting()
table_name = 'table_name'
df = pd.read_excel('local 위', header=0,sheet_name='mptestdb')
df = df.replace(0, np.nan).dropna()
insert_into_sql(table_name, df)

