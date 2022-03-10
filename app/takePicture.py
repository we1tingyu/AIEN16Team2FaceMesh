import cv2
from datetime import datetime
from faceMeshProjectForFACE_OVAL import FaceMeshDetector

cap = None
msg =''
msgnew =''  
def faceCondition(photograph):          
    global msgnew   
    global msg  
    global cap  
    errorMsg =''
    if photograph =='啟動':
        cap = cv2.VideoCapture(0) #建立一個 VideoCapture 物件 0號設備                
        while cap.isOpened(): #迴圈讀取每一幀

            k = cv2.waitKey(1) & 0xFF  #每幀資料延時 1ms，延時不能為 0，否則讀取的結果會是靜態幀
            
            # if k == ord('q'): #若檢測到按鍵 ‘q’，退出q
            #     break   
            
            ret, img = cap.read()

            if not ret:
                print("error")
                break            
            
            img=cv2.flip(img,1)  # 解決鏡頭左右相反的問題
            # cv2.imshow("CameraLive (^.<) ",img)  #視窗顯示，顯示名為 CameraLive

            
            # 傳送至前端
            # todo 將msg劃入img中 傳送至前端
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
        ret, img = cap.read()
        img=cv2.flip(img,1)  # 解決鏡頭左右相反的問題
        
        # 傳送至前端
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        print('===========================')
        print(msg)
                    
        # print('請輸入姓名:')
        # str=input()
        time=datetime.now().strftime('%Y%m%d%H%M%S')
        cv2.imencode('.jpg', img)[1].tofile("./app/static/images/"+ time +".jpg")
        print(cap.get(3)); #得到長寬
        print(cap.get(4))
        print("success to save:"+ time +".jpg")
        print("-------------------------")    

        cap.release()  #釋放攝像頭
        # cv2.destoryAllWindows() #刪除建立的全部視窗                      

      
    
    else :
        errorMsg = '請啟動相機'
        print('請啟動相機')

    
# 執行func.
# faceCondition()
        

        

   
    


    



