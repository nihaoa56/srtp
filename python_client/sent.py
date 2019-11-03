import cv2
import sys
import os
import time
import requests

# 定义图片保存位置
# path = os.getcwd() 
url = 'http://47.93.246.19/doFile/doFile.php'


def pic_sent():
    time.sleep(1)
    print(time.ctime())
    files = {'file123': ('person.jpg', open('python_client/test.png', 'rb'))
        }  # 显式的设置文件名
    r = requests.post(url, files=files)
    print(r.text)
while (1):
	pic_sent()