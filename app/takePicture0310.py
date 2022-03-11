# coding:utf-8
import cv2
from datetime import datetime
from faceMeshProjectForFACE_OVAL import FaceMeshDetector
from PIL import Image, ImageFont, ImageDraw
import numpy
import os

# 將txt畫至img
def add_txt_to_image(img, txt='', position=(10, 40)):
   
    # 只能畫英文到圖上
    # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
    # cv2.putText(img, str,  position ,  font ,  1, (0, 255, 255), 1, cv2.LINE_AA)

    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # font = ImageFont.load_default()
    font = ImageFont.truetype(
        os.path.abspath(os.path.dirname(__file__)) +
        '/static/fonts/JasonHandwriting4.ttf', 30)
    # font = ImageFont.truetype(r'C:\Users\Student\Desktop\git-facemesh\AIEN16Team2FaceMesh\app\static\fonts\JasonHandwriting4.ttf', 50)

    # 輸出內容
    # str = msgnew
    # 需要先把輸出的中文字元轉換成Unicode編碼形式

    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, txt, 'blue', font)
    # 使用PIL中的save方法儲存圖片到本地
    # img_PIL.save('02.jpg', 'jpeg')

    # 轉換回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)


#取得臉周長，並輸出txt
txt=''
def get_txt(img):    
    distance = FaceMeshDetector(maxFaces=10).findFaceMesh(
        img.copy(),
        drawFaceLms=True,
        drawID=False,
        drawFortuneTelling="臉部外框",
        takePicture=True)
    # print(distance)

    if type(distance) is float:
        distance = int(distance)
        
        if distance < 650:
            txt = '請將臉部靠近鏡頭'
        elif distance > 900:
            txt = '幹你老蘇哩 靠太近啦!'
        else:
            txt = '已符合測量條件,請按下拍照'
    else:
            txt = '人哩?'
    
    return txt

def faceCondition(camera_status,
                  # msg='',
                  # msgnew='',                  
                  ):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #建立一個 VideoCapture 物件 0號設備

    global txt        
   
    print(camera_status) 
    
    flag=True
    if camera_status == "拍照" and txt=='已符合測量條件,請按下拍照':
        flag = False
    elif camera_status == "啟動" or camera_status == "拍照":        
        flag = True
    

    while cap.isOpened()  :  #迴圈讀取每一幀

        # if k == ord('q'): #若檢測到按鍵 ‘q’，退出q
        #     break
        ret, img = cap.read()

        if not ret:
            print("camera byebye")
            continue

        img = cv2.flip(img, 1)  # 解決鏡頭左右相反的問題   

        txt= get_txt(img)
        img = add_txt_to_image(img, txt)        
        
        if not flag :

            time = datetime.now().strftime('%Y%m%d%H%M%S')
            cv2.imencode('.jpg',
                        img)[1].tofile("./app/static/images/" + time + ".jpg")
            #得到長寬
            print(cap.get(3))        
            print(cap.get(4))
            print("success to save:" + time + ".jpg")
            print("-------------------------")
            cv2.destroyAllWindows()  #刪除建立的全部視窗
            
            return img
        
        elif flag :    
            # 傳送至前端
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

   


# 執行func.
# faceCondition()

# todo : 1. 第二次啟動後的拍照 失敗
#        2.非適當距離的時候按拍照發生後 下次無法拍照
