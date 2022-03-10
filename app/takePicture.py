# coding:utf-8
import imp
import cv2
from datetime import datetime
from faceMeshProjectForFACE_OVAL import FaceMeshDetector
from PIL import Image,ImageFont,ImageDraw
import numpy
import os

cap = None
msg =''
msgnew =''  
def faceCondition(photograph):          
    global msgnew   
    global msg  
    global cap  
    errorMsg =''
    
    
          
    if photograph =='啟動':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #建立一個 VideoCapture 物件 0號設備              
        while cap.isOpened(): #迴圈讀取每一幀

            k = cv2.waitKey(1) & 0xFF  #每幀資料延時 1ms，延時不能為 0，否則讀取的結果會是靜態幀
            
            # if k == ord('q'): #若檢測到按鍵 ‘q’，退出q
            #     break   
            
            ret, img = cap.read()

            if not ret:
                print("camera byebye")
                break            
            
            img=cv2.flip(img,1)  # 解決鏡頭左右相反的問題
            # cv2.imshow("CameraLive (^.<) ",img)  #視窗顯示，顯示名為 CameraLive

            # 只能畫英文到圖上
            # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            # cv2.putText(img, str,  position ,  font ,  1, (0, 255, 255), 1, cv2.LINE_AA)
            
            img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # font = ImageFont.load_default()
            font = ImageFont.truetype(os.path.abspath(os.path.dirname(__file__))+'/static/fonts/JasonHandwriting4.ttf', 50)
            # font = ImageFont.truetype(r'C:\Users\Student\Desktop\git-facemesh\AIEN16Team2FaceMesh\app\static\fonts\JasonHandwriting4.ttf', 50)
            

            # 文字輸出位置
            position = (10, 40)
            # 輸出內容
            str = msgnew
            # 需要先把輸出的中文字元轉換成Unicode編碼形式           
            
           
            draw = ImageDraw.Draw(img_PIL)
            draw.text(position, str, 'blue',font )
            # 使用PIL中的save方法儲存圖片到本地
            # img_PIL.save('02.jpg', 'jpeg')

            # 轉換回OpenCV格式
            img = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)
            
            
            # 傳送至前端
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            #取得臉周長
            detector = FaceMeshDetector(maxFaces=10)
            distance = detector.findFaceMesh(img, drawFaceLms=True, drawID=False, drawFortuneTelling="臉部外框",takePicture=True)  
            # print(distance)

            
            if type(distance) is float:  
                distance= int(distance)
                
                if distance<650 :
                    msgnew ='請將臉部靠近鏡頭'
                elif distance>900:
                    msgnew ='幹你老蘇哩 靠太近啦!'
                else:
                    msgnew ='已符合測量條件,請按下拍照'                 
                 

                # if msgnew =='已符合測量條件,請按S拍照' and  k == ord('s'):  #若檢測到按鍵 ‘s’，列印字串
                 
                    

                    # if msgnew =='已符合測量條件,請按下拍照' and k=='拍照':  #若檢測到前端傳來的訊號，則執行
                    #     print('請輸入姓名:')
                    #     str=input()
                    #     time=datetime.now().strftime('%Y%m%d%H%M%S')
                    #     cv2.imencode('.jpg', img)[1].tofile("C:/Users/Student/Desktop/"+ time +" "+ str + ".jpg")
                    #     print(cap.get(3)); #得到長寬
                    #     print(cap.get(4))
                    #     print("success to save:"+ time +" "+str+".jpg")
                    #     print("-------------------------")                                

                    # elif k == ord('q'): #若檢測到按鍵 ‘q’，退出q
                    #     break

            
            else :              
                msgnew ='人哩?'

            
            if msg!=msgnew :
                msg=msgnew
                print(msg)
        
            # if photograph =='拍照'and msg =='已符合測量條件,請按下拍照':
            #     break
                
                  
        
    
    elif photograph =='拍照' and msg =='已符合測量條件,請按下拍照':        
        # print("一兵要求------------------------------------")
        ret, img = cap.read()
        # print(ret)
        img=cv2.flip(img,1)  # 解決鏡頭左右相反的問題
        
        # 傳送至前端
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # print('===========================')
        # print(msg)
                    
        # print('請輸入姓名:')
        # str=input()
        time=datetime.now().strftime('%Y%m%d%H%M%S')
        cv2.imencode('.jpg', img)[1].tofile("./app/static/images/"+ time +".jpg")
        print(cap.get(3)); #得到長寬
        print(cap.get(4))
        print("success to save:"+ time +".jpg")
        print("-------------------------")    

        cap.release()  #釋放攝像頭
        cv2.destroyAllWindows() #刪除建立的全部視窗   
                      

      
    
    else :
        # errorMsg = '請啟動相機'
        print('請將臉調整至適當位子')

    
# 執行func.
# faceCondition()
        
# todo : 1. 第二次啟動後的拍照 失敗 
#        2.非適當距離的時候按拍照發生後 下次無法拍照
        

   
    


    



