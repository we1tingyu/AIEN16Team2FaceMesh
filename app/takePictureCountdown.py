# coding:utf-8
# 需要先把輸出的中文字元轉換成Unicode編碼形式
import cv2
import time
from datetime import datetime,timedelta 
import threading
from flask import Flask
from PIL import Image, ImageFont, ImageDraw
import numpy
import os
from faceMeshProjectForFACE_OVAL import FaceMeshDetector

# 創建一個lock對象: lock = threading.Lock()
# 獲取使用lock: lock.acquire()
# 解除釋放lock: lock.release()
# 釋放前其他Thread(執行緒)無法執行
# lock = threading.Lock()


# 將txt畫至img
def add_txt_to_image(img, txt='', xy=(10, 40)):
   
    # 只能畫英文到圖上
    # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
    # cv2.putText(img, str,  position ,  font ,  1, (0, 255, 255), 1, cv2.LINE_AA)

    img_PIL = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))

    if txt == '請將臉部靠近鏡頭' : 
        shape = [(xy[0]-3, xy[1]-1), (xy[0]+250, xy[1]+40)] 
        wordColor = '#ffff2b'
    elif txt == '乾你老蘇哩 靠太近啦!' :
        shape = [(xy[0]-1, xy[1]-1), (xy[0]+300, xy[1]+40)] 
        wordColor = '#fb0301'
    elif txt == '臉部稍微歪斜 請擺正' :
        shape = [(xy[0]-1, xy[1]-1), (xy[0]+300, xy[1]+40)] 
        wordColor = '#ffff2b'
    elif txt == '霍金叔叔 您回來啦!' :
        shape = [(xy[0]-1, xy[1]-1), (xy[0]+275, xy[1]+40)] 
        wordColor = '#fb0301'
    elif txt == '人哩?' :
        shape = [(xy[0]-1, xy[1]-1), (xy[0]+80, xy[1]+40)] 
        wordColor = '#ffffff'
    else :
        shape = [(xy[0]-1, xy[1]-1), (xy[0]+210, xy[1]+40)] 
        wordColor = '#74d56a'

    
    draw = ImageDraw.Draw(img_PIL)   
    draw.rectangle(shape, fill ="#000000", outline ="#0136ae", width=2 ) 


    # font = ImageFont.load_default()
    font = ImageFont.truetype(
        os.path.abspath(os.path.dirname(__file__)) +'/static/fonts/JasonHandwriting4.ttf', 30)
    # font = ImageFont.truetype(r'C:\Users\Student\Desktop\git-facemesh\AIEN16Team2FaceMesh\app\static\fonts\JasonHandwriting4.ttf', 50)
      
   
    draw.text(xy, txt, wordColor , font)
    # 使用PIL中的save方法儲存圖片到本地
    # img_PIL.save('02.jpg', 'jpeg')

    # 轉換回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)


#取得臉周長及歪斜，並輸出txt
txt=''
def get_txt(img):    
    distance = FaceMeshDetector(maxFaces=1).findFaceMesh(
        img.copy(),
        drawFaceLms=True,
        drawID=False,
        drawFortuneTelling="臉部外框",
        takePicture=True)
    # print(distance)
    FaceX = FaceMeshDetector(maxFaces=1).findFaceMesh(
        img.copy(),
        drawFaceLms=True,
        drawID=False,
        drawFortuneTelling="歪臉判定",
        takePicture=True)
    # print(angle)
    
    # 依主程式回傳之臉周長 判斷使用者與攝影機之間的距離
    if type(distance) is float:
        distance = int(distance)
        
        if distance < 650:
            txt = '請將臉部靠近鏡頭'
        elif distance > 900:
            txt = '乾你老蘇哩 靠太近啦!'
        else:
            if FaceX >30 and FaceX <= 60:
                txt = "臉部稍微歪斜 請擺正"
            elif  FaceX > 60 :
                txt = "霍金叔叔 您回來啦!"
            else : 
                txt = '已符合測量條件,請按下拍照'
    else:
            txt = '人哩?'
    
    return txt

# 倒數計時器(沒用到、寫爽的)
def countdown(num_of_secs):
    
    while num_of_secs:
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        time.sleep(1)
        num_of_secs -= 1
        return min_sec_format
   

# 將照片儲存起來
def savePicture(img,cap): 
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    # 可儲存中文檔案名稱
    cv2.imencode('.jpg',
                img)[1].tofile("./app/static/images/" + time + ".jpg") 

    cv2.imencode('.jpg',
                img)[1].tofile("./app/static/images/" + 'Thelatestphotos' + ".jpg")
                                   
    #得到長寬
    print(cap.get(3))        
    print(cap.get(4))
    print("success to save:" + time + ".jpg")
    print("-------------------------")
    cap.release()
    cv2.destroyAllWindows()  #刪除建立的全部視窗

    return img   
        
# 拍照主程式    
def streamlive(camera_status):
    # global lock
    # global cap
    global txt
    # global img

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #建立一個 VideoCapture 物件 0號設備

    time_flag = False

    while cap.isOpened() and cv2.waitKey(1) :  # cap.isOpened()確認攝影機有打開  # 迴圈讀取每一幀
        
        # if k == ord('q'): #若檢測到按鍵 ‘q’，退出q
        #     break
        ret, img = cap.read()    # ret:True/False

        if not ret:
            print("camera byebye")
            break
        
        # with lock :

        img = cv2.flip(img, 1)  # 解決鏡頭左右相反的問題   
        
        txt= get_txt(img)
        # imgtxt = add_txt_to_image(img, txt)
        
        

        if  txt=='已符合測量條件,請按下拍照' : 
                       
            if not time_flag:   # 僅只在第一次進入迴圈 才會進入
                cTime = time.time()
                time_flag = True
            
            pTime = time.time()
            s_format = "於"+'{:02d}:{:02d}'.format(0, 3-int(pTime - cTime)) + "秒後拍照"
            imgtxt=add_txt_to_image(img,s_format,xy=(20,40))

            
            if  int(pTime - cTime) == 3:
                imgtxt=savePicture(img,cap) 
                
        else:
            time_flag = False
            imgtxt = add_txt_to_image(img, txt)                        
        
        
                
        # 傳送至前端
        frame = cv2.imencode('.jpg', imgtxt)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 若偵測到鏡頭沒開，則回傳最後一張img至前端
    if not cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()  #刪除建立的全部視窗
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

              
               
    
