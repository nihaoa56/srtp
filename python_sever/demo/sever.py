from flask import Flask, request
import pymysql

raw_data_path = "/home/sun/桌面/srtp-git/raw_data/"
aim_path = "/home/sun/桌面/srtp-git/page/images/"




def mysql_insert(dataPath, realName, dataDescripe, nickName, password, CZ, email, tips):
    conn = pymysql.connect(host="127.0.0.1", user="root",
                       password="a1239588540", database="srtp", charset="utf8")
    # 打开标签对象s
    cursor = conn.cursor()
    sql = "INSERT INTO data_stable(dataPath,realName,dataDescripe,nickName,password,CZ,email,tips) VALUES (%s,%s,%s,%s,%s,%s,%s,%s);"
    # 执行SQL语句
    cursor.execute(sql, [dataPath,realName,dataDescripe,nickName,password,CZ,email,tips])
    # 提交事务
    conn.commit()
    # 关闭光标对象
    cursor.close()
    # 关闭数据库连接
    conn.close()

def create_line_pic():
    print('create line over')

def create_cluster_list():
    print('create cluster list over !')

def create_NN_classfiter_list():
    print("create NN_classfiter list over !")

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/data_adder', methods=['POST','GET'])
def register():
    print(request.form)
    dataPath = request.form['identity']    #属性 姓名地址.
    dataDescripe = request.form['data_description'] #描述
    realName = request.form['realName'] 
    nickName = request.form['loginName']
    password = request.form['password']
    # data_password2 = request.form['password2']
    CZ = request.form['cz']
    email = request.form['email']
    tips = request.form['tips']

    mysql_insert(dataPath, realName, dataDescripe, nickName, password, CZ, email, tips)

    return 'success!'

if __name__ == '__main__':
    app.run(port=8838,debug=True)