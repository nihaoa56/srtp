from flask import Flask, request
import pymysql
import os
import shutil  # 文件移动操作

raw_data_path = "/home/sun/桌面/srtp-git/raw_data/"
# 备选 选项   matrix_difference    peak_iteration     threshold_value_iteration
aim_path = "/home/sun/桌面/srtp-git/page/images/"
# 矩阵差分                          峰值迭代                    阈值分割


def get_filename_inDir(dirPath):  # 得到目录下所有非目录文件
    file_name = list()
    for root, dirs, files in os.walk(dirPath):
        file_name = files
    return file_name


def delate_file_inDir(dirPath):
    file_name = get_filename_inDir(dirPath)
    for file in file_name:
        file_path = dirPath + file
        os.remove(file_path)
    return "remove over"


def copy_file_to(raw_data, raw_data_path):
    file = {"peak_interation": [], "threshold_value_iteration": [],
            "matrix_difference": []}  # 定义文件分类
    for name in raw_data:
        if name == "1.jpg" or name == "2.jpg":
            file["peak_interation"].append(name)
        elif name == "3.jpg" or name == "4.jpg":
            file["threshold_value_iteration"].append(name)
        elif name == "5.jpg" or name == "6.jpg":
            file["matrix_difference"].append(name)
        else:
            raise Exception("logic Deletion")  # 逻辑缺少,错误跳出
    srcfile_list = list()
    dstfile_list = list()
    for key, value in file.items():
        for file in value:
            srcfile = raw_data_path + file
            dstfile = aim_path + key + "/" + file
            
            # print("srcfile", srcfile)
            srcfile_list.append(srcfile)
            # print("dstfile", dstfile)
            dstfile_list.append(dstfile)
    # print(srcfile_list)
    # print("-" * 50)
    # print(dstfile_list)
    for number in range(len(srcfile_list)):
        print(srcfile_list[number])
        print(dstfile_list[number])
        shutil.copyfile(srcfile_list[number],dstfile_list[number])
           


def mysql_insert(dataPath, realName, dataDescripe, nickName, password, CZ, email, tips):
    conn = pymysql.connect(host="127.0.0.1", user="root",
                           password="a1239588540", database="srtp", charset="utf8")
    # 打开标签对象s
    cursor = conn.cursor()
    sql = "INSERT INTO data_stable(dataPath,realName,dataDescripe,nickName,password,CZ,email,tips) VALUES (%s,%s,%s,%s,%s,%s,%s,%s);"
    # 执行SQL语句
    cursor.execute(sql, [dataPath, realName, dataDescripe,
                         nickName, password, CZ, email, tips])
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


@app.route('/data_adder', methods=['POST', 'GET'])
def register():
    print(request.form)
    dataPath = request.form['identity']  # 属性 姓名地址.
    dataDescripe = request.form['data_description']  # 描述
    realName = request.form['realName']
    nickName = request.form['loginName']
    password = request.form['password']
    # data_password2 = request.form['password2']
    CZ = request.form['cz']
    email = request.form['email']
    tips = request.form['tips']

    if dataPath[-1] != "/":  # 格式统一
        dataPath = dataPath + "/"
    else:
        pass

    # 删除目标区旧文件
    # delate_file_inDir(aim_path + "peak_interation/")
    delate_file_inDir(aim_path + "threshold_value_iteration/")
    delate_file_inDir(aim_path + "matrix_difference/")

    file_nameInDir = get_filename_inDir(dataPath)

    copy_file_to(file_nameInDir, dataPath)

    # mysql_insert(dataPath, realName, dataDescripe, nickName,
    #              password, CZ, email, tips)  # 存入数据库

    return 'success!'


if __name__ == '__main__':
    app.run(port=8838, debug=True)
