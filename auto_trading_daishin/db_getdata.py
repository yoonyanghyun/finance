import pymysql
import pandas as pd
import yfinance as yf
import os

yf.pdr_override()

# ==============================================================================
# ==============================================================================
def db_ssh_tunneling():
    path_ssh = 'local ssh 위치'
    os.chdir(path_ssh)
    os.system('ssh.exe')
    path_pem = 'ssh key file'
    cmd = 'ssh"'%path_pem
    os.system('start cmd /k' + cmd)

# ==============================================================================
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


def db_read(sql):
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall())
    return df

db_ssh_tunneling()
db_connecting()

sql = 'select * from table_name '
sql = sql+' where condition1 and condition2'
df = db_read(sql)

dt_rebal = df.created_at.drop_duplicates().sort_values().tolist()[-1]
dt_befor = df.created_at.drop_duplicates().sort_values().tolist()[-2]

df_rebal = df[df.created_at == dt_rebal][['symbol','weight']]
df_befor = df[df.created_at == dt_befor][['symbol','weight']]

df_rebal.rename(columns = {'weight':dt_rebal}, inplace = True)
df_befor.rename(columns = {'weight':dt_befor}, inplace = True)

df_result = pd.merge(df_befor, df_rebal, left_on='symbol', right_on='symbol', how='outer')
df_result.fillna(0, inplace=True)


