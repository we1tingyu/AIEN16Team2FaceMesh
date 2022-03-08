import cv2
from faceMeshProjectForFACE_OVAL import FaceMeshDetector


def faceCondition():
    cap = cv2.VideoCapture(0) #建立一個 VideoCapture 物件 0號設備
    msg =''
    while cap.isOpened(): #迴圈讀取每一幀
        if k == ord('q'): #若檢測到按鍵 ‘q’，退出q
            break   
        ret, img = cap.read()
        if not ret:
            print("error")
            break
        
        
        img=cv2.flip(img,1)  # 解決鏡頭左右相反的問題
        cv2.imshow("CameraLive (^.<) ",img)  #視窗顯示，顯示名為 CameraLive
        k = cv2.waitKey(1) & 0xFF  #每幀資料延時 1ms，延時不能為 0，否則讀取的結果會是靜態幀

        detector = FaceMeshDetector(maxFaces=10)
        distance = detector.findFaceMesh(img, drawFaceLms=True, drawID=False, drawFortuneTelling=1,takePicture=True)  
        # print(type(distance))
        if type(distance) is float:  
            distance= int(distance)
            # print(distance)

            msgnew =''
            if distance<700 :
                msgnew ='請將臉部靠近鏡頭'
            elif distance>900:
                msgnew ='幹你老蘇哩 靠太近啦!'
            else:
                msgnew ='已符合測量條件,請按S拍照'
            
            if msg!=msgnew:
                msg==msgnew
                print(msg)

            if msgnew =='已符合測量條件,請按S拍照' and k == ord('s'):  #若檢測到按鍵 ‘s’，列印字串
                print('請輸入姓名:')
                str=input()
                cv2.imencode('.jpg', img)[1].tofile("C:/Users/Student/Desktop/"+ str + ".jpg")
                print(cap.get(3)); #得到長寬
                print(cap.get(4))
                print("success to save:"+str+".jpg")
                print("-------------------------")                                

            elif k == ord('q'): #若檢測到按鍵 ‘q’，退出q
                break

        
        else : 
            print('人哩?') 
      
            
    cap.release()  #釋放攝像頭
    cv2.destoryAllWindows() #刪除建立的全部視窗

# 執行func.
faceCondition()
        

        

   
    


    

# img, faces, distance, sum = FaceMeshDetector.findFaceMesh(img, drawFaceLms=True, drawID=False, drawFortuneTelling=1)
# print(sum)

