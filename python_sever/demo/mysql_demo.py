import pymysql

# 连接数据库
conn = pymysql.connect(host="127.0.0.1", user="root",
                       password="a1239588540", database="srtp", charset="utf8")
cursor = conn.cursor()

# sql = """
# CREATE TABLE data_stable (
# id INT auto_increment PRIMARY KEY,
# dataPath varchar(50) not null,
# realName CHAR(10) NOT NULL,
# dataDescripe TEXT,
# nickName VARCHAR(20) NOT NULL,
# password int(20) NOT NULL,
# CZ INT,
# email VARCHAR(30),
# tips TEXT
# )ENGINE=innodb DEFAULT CHARSET=utf8;
# """
dataPath = "1"
realName = "1"
dataDescripe = "1"
nickName = "1"
password = "1"
CZ = "1"
email = "1"
tips = "1"
sql = "INSERT INTO data_stable(dataPath,realName,dataDescripe,nickName,password,CZ,email,tips) VALUES (%s,%s,%s,%s,%s,%s,%s,%s);"

# 执行SQL语句
cursor.execute(sql, [dataPath,realName,dataDescripe,nickName,password,CZ,email,tips])
# 提交事务
conn.commit()

# 关闭光标对象
cursor.close()
# 关闭数据库连接
conn.close()
