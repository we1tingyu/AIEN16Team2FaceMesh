
# AIEN16 第二組 臉部偵測Sample code --> git clone from google mediapipe website.
# google MediaPipe - AI機器學習套件介紹, 它的套件包括: 
# 臉部偵測, 臉部特徵立體偵測, 手部關節運動追蹤, 人像與背景的分割, 髮型特徵分割, 人體立體運動姿勢追蹤, 物件辨識, 
# 臉/身體/支幹整體偵測, 3D物件偵測.
# https://google.github.io/mediapipe/
# google MediaPipe github 
# https://github.com/google/mediapipe
#

import cv2
import mediapipe as mp
import time
import math
import threading
import facialFeatures
 

# 創建一個lock對象: lock = threading.Lock()
# 獲取使用lock: lock.acquire()
# 解除釋放lock: lock.release()
# 釋放前其他Thread(執行緒)無法執行
lock = threading.Lock()


BLUE_XXX = (255, 106 , 90)
BLACK = (0, 0, 0)
GREEN = (0, 106 , 90)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# COLOR 可以由前端按按鈕切換
DEFAULT_COLOR = BLUE

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=10, minDetectionCon=False, minTrackCon=0.5):
        
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        # 畫出臉上的線條
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        # there are 468 points.
        self.drawSpec = self.mpDraw.DrawingSpec(color=(30,144,255), thickness=1, circle_radius=1) 

    def findFaceMesh(self, img, drawFaceLms=True, drawID=False, drawFortuneTelling="臉部特徵網格圖",takePicture=False):
        # 左右相反，for camera
        #self.imgRGB = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB) # opposite right/left for actual face sync-up detection when face turns left/right.
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        # put faces[]/distance{} here to avoid error"local variable referenced before assignment" 
        faces = []
        distance = []

        ih, iw, ic = img.shape

        FACE_OVAL=facialFeatures.FACE_OVAL
        READ_FACE=facialFeatures.READ_FACE

        EDGE=facialFeatures.EDGE

        THREE_COURT=facialFeatures.THREE_COURT
        Beauty_Corner=facialFeatures.Beauty_Corner

        three_court_y = []
        three_court_ratio = []

        FIVE_EYE=facialFeatures.FIVE_EYE
        five_eye_x = []
        five_eye_ratio = []
        sum=0  
        # multi_face_landmarks :臉上作完正規化的xyz座標
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:                
                if drawFaceLms:
                    # image=img, landmark_list=faceLms, connections=self.mpFaceMesh.FACEMESH_TESSELATION, 
                    # landmark_drawing_spec=self.drawSpec,
                    # connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_tesselation_style().
                    # 臉上的各點及其連線
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                                self.drawSpec, self.mpDrawingStyles.get_default_face_mesh_tesselation_style())
                    # image=img, landmark_list=faceLms, connections=self.mpFaceMesh.FACEMESH_CONTOURS, 
                    # landmark_drawing_spec=self.drawSpec,
                    # connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_contours_style().
                    # goolge劃出的範例線條
                    # self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                    #                             self.drawSpec, self.mpDrawingStyles.get_default_face_mesh_contours_style())

                    # 取得最斯個
                    top_ID = EDGE['top']
                    bottom_ID = EDGE['bottom']
                    left_ID = EDGE['left']
                    right_ID = EDGE['right']

                    top_y = faceLms.landmark[top_ID].y*ih
                    bottom_y = faceLms.landmark[bottom_ID].y*ih
                    left_x = faceLms.landmark[left_ID].x*iw
                    right_x = faceLms.landmark[right_ID].x*iw
                
                # 臉部特徵網格圖, 什麼都不畫
                if drawFortuneTelling == "臉部特徵網格圖":
                    pass
                
                # 臉部外框
                elif drawFortuneTelling == "臉部外框":
                    # draw specific IDs for fortune telling(畫出上面自行定義的各點)
                    # for idx1,value in enumerate(FACE_OVAL):
                    for idx1,ff in enumerate(FACE_OVAL):
                        sum=0
                        for idx2,value in enumerate(FACE_OVAL[idx1]):

                            startID, endID = value
                            # print(f'startID:{startID} endID:{endID} ')
                            # faceID to xyz                            
                            # ih, iw, ic = img.shape
                            lm_start = faceLms.landmark[startID]
                            # x, y, z = int(lm_start.x*iw), int(lm_start.y*ih), int(lm_start.z*ic)
                            x1, y1, z1 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                            # 起點的 2D int 座標 (給 cv2 用)
                            startAddress2D = int(x1), int(y1)
                            lm_end = faceLms.landmark[endID]
                            x2, y2, z2 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic
                            # 終點的 2D int 座標 (給 cv2 用)
                            endAddress2D = int(x2), int(y2)
                            # print(f'startAddress:{startAddress},endAddress:{endAddress}')

                            # draw specific line user defined (只畫線, 不算距離)
                            self.drawSpecificLine(img, startAddress2D, endAddress2D)

                            # 起點的 3D int 座標
                            startAddress3D = x1, y1, z1
                            # 終點的 3D int 座標
                            endAddress3D = x2, y2, z2

                            # Euclaidean Distance 計算出距離
                            # 計算 2D 距離
                            # lineDistance = self.euclaideanDistance(startAddress2D, endAddress2D)
                            # 計算 3D 距離
                            lineDistance = self.euclaideanDistance3D(startAddress3D, endAddress3D)
                            # lineDistance = int(lineDistance)
                            # print(f'lineDistance (單站里程):{lineDistance}')

                            sum+=lineDistance
                            
                            # print(f'FACE_OVAL (目前累積里程):{sum}')    
                            # append into distance[]
                            distance.append(lineDistance)

                    # print('------')
                    # print(f'FACE_OVAL (總里程):{sum}')

                    # 拍照用
                    if takePicture :
                        return sum
                        
                    print(f'臉的周長:{sum}')
                    print('------------')

                # 臉部面相算命特徵圖, 五官
                elif drawFortuneTelling == "臉部面相算命特徵圖":
                    # draw specific IDs for fortune telling(畫出上面自行定義的各點)
                    # READ_FACE=facialFeatures.READ_FACE
                    for idx1,ff in enumerate(READ_FACE):
                        sum=0
                        for idx2,value in enumerate(READ_FACE[idx1]):
                            startID, endID = value
                            # print(f'startID:{startID} endID:{endID} ')
                            # faceID to xyz                            
                            # ih, iw, ic = img.shape
                            lm_start = faceLms.landmark[startID]
                            # x, y, z = int(lm_start.x*iw), int(lm_start.y*ih), int(lm_start.z*ic)
                            x1, y1, z1 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                            # 起點的 2D int 座標 (給 cv2 用)
                            startAddress2D = int(x1), int(y1)
                            lm_end = faceLms.landmark[endID]
                            x2, y2, z2 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic
                            # 終點的 2D int 座標 (給 cv2 用)
                            endAddress2D = int(x2), int(y2)
                            # print(f'startAddress:{startAddress},endAddress:{endAddress}')

                            # draw specific line user defined (只畫線, 不算距離)
                            self.drawSpecificLine(img, startAddress2D, endAddress2D, BLACK)
                            
                            # 起點的 3D int 座標
                            startAddress3D = x1, y1, z1
                            # 終點的 3D int 座標
                            endAddress3D = x2, y2, z2

                            # Euclaidean Distance 計算出距離
                            # 計算 2D 距離
                            # lineDistance = self.euclaideanDistance(startAddress2D, endAddress2D)
                            # 計算 3D 距離
                            lineDistance = self.euclaideanDistance3D(startAddress3D, endAddress3D)

                            # lineDistance = self.drawSpecificLine(img, startAddress, endAddress)
                            # lineDistance = int(lineDistance)
                            sum+=lineDistance
                            # print(lineDistance)
                            # print(sum)
                        # append into distance[]                           
                        distance.append(sum)                     
                    if distance:
                        print(f'RIGHT_EYEBROW:{distance[0]}, LEFT_EYEBROW:{distance[1]}, RIGHT_EYE:{distance[2]}, LEFT_EYE{distance[3]}, NOSE_LENGTH:{distance[4]}, NOSE_WIDTH:{distance[5]}, FOREHEAD:{distance[6]}, PHILTRUM:{distance[7]}, MOUTH:{distance[8]}')
                        print('------------')

                # 三庭
                elif drawFortuneTelling == "三庭":
                    for idx1,value in enumerate(THREE_COURT):
                        ID = value

                        lm = faceLms.landmark[ID]
                        # x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                        x, y, z = lm.x*iw, lm.y*ih, lm.z*ic

                        # 起點的 2D int 座標 (給 cv2 用)
                        startAddress2D = int(left_x), int(y)
                        # 終點的 2D int 座標 (給 cv2 用)
                        endAddress2D = int(right_x), int(y)

                        self.drawSpecificLine(img, startAddress2D, endAddress2D, RED)

                        three_court_y.append(y)

                    if three_court_y:
                        print(f'由上到下的 y 座標分別是 y1:{three_court_y[0]}, y2:{three_court_y[1]}, y3:{three_court_y[2]}, y4:{three_court_y[3]}')

                        total_y = three_court_y[3] - three_court_y[0]
                        
                        print(f'上到下的 y 距離:{total_y}')

                        for i in range(len(THREE_COURT) - 1):
                            y_distance = three_court_y[i+1] - three_court_y[i]
                            ratio = y_distance / total_y
                            # print(ratio)
                            three_court_ratio.append(ratio)

                        print(f'三庭(上到下)比例為-> {three_court_ratio[0]}:{three_court_ratio[1]}:{three_court_ratio[2]}')
                        print('------------')
                
                # 五眼
                elif drawFortuneTelling == "五眼":
                    for idx1,value in enumerate(FIVE_EYE):
                        # print(value)
                        ID = value

                        lm = faceLms.landmark[ID]
                        # x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                        x, y, z = lm.x*iw, lm.y*ih, lm.z*ic

                        # 起點的 2D int 座標 (給 cv2 用)
                        startAddress2D = int(x), int(top_y)
                        # 終點的 2D int 座標 (給 cv2 用)
                        endAddress2D = int(x), int(bottom_y)

                        self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                        five_eye_x.append(x)

                    if five_eye_x:
                        print(f'由左到右的 x 座標分別是 x1:{five_eye_x[0]}, x2:{five_eye_x[1]}, x3:{five_eye_x[2]}, x4:{five_eye_x[3]}, x5:{five_eye_x[4]}, x6:{five_eye_x[5]}')

                        total_x = five_eye_x[5] - five_eye_x[0]
                        
                        print(f'左到右的 x 距離:{total_x}')

                        for i in range(len(FIVE_EYE) - 1):
                            x_distance = five_eye_x[i+1] - five_eye_x[i]
                            ratio = x_distance / total_x
                            # print(ratio)
                            five_eye_ratio.append(ratio)

                        print(f'五眼(左到右)比例為-> {five_eye_ratio[0]}:{five_eye_ratio[1]}:{five_eye_ratio[2]}:{five_eye_ratio[3]}:{five_eye_ratio[4]}')
                        print('------------')

                elif drawFortuneTelling == 5:
                    for idx1,ff in enumerate(Beauty_Corner):
                        sum=0
                        for idx2,value in enumerate(Beauty_Corner[idx1]):
                            startID, endID = value
                            # print(f'startID:{startID} endID:{endID} ')
                            # faceID to xyz                            
                            ih, iw, ic = img.shape
                            lm_start = faceLms.landmark[startID]
                            # x, y, z = int(lm_start.x*iw), int(lm_start.y*ih), int(lm_start.z*ic)
                            x1, y1, z1 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                            # 起點的 2D int 座標 (給 cv2 用)
                            startAddress2D = int(x1), int(y1)
                            lm_end = faceLms.landmark[endID]
                            x2, y2, z2 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic
                            # 終點的 2D int 座標 (給 cv2 用)
                            endAddress2D = int(x2), int(y2)
                            # print(f'startAddress:{startAddress},endAddress:{endAddress}')

                            # draw specific line user defined (只畫線, 不算距離)
                            self.drawSpecificLine(img, startAddress2D, endAddress2D, BLACK)
                            
                            # 起點的 3D int 座標
                            startAddress3D = x1, y1, z1
                            # 終點的 3D int 座標
                            endAddress3D = x2, y2, z2

                            # Euclaidean Distance 計算出距離
                            # 計算 2D 距離
                            # lineDistance = self.euclaideanDistance(startAddress2D, endAddress2D)
                            # 計算 3D 距離
                            lineDistance = self.angle(startAddress2D, endAddress2D)

                            # lineDistance = self.drawSpecificLine(img, startAddress, endAddress)
                            # lineDistance = int(lineDistance)
                            # sum+=lineDistance
                            # print(lineDistance)
                            # print(sum)
                        # append into distance[]                           
                        distance.append(sum)                     
                    # if distance:
                    #     print(f'RIGHT_EYEBROW:{distance[0]}, LEFT_EYEBROW:{distance[1]}, RIGHT_EYE:{distance[2]}, LEFT_EYE{distance[3]}, NOSE_LENGTH:{distance[4]}, NOSE_WIDTH:{distance[5]}, FOREHEAD:{distance[6]}, PHILTRUM:{distance[7]}, MOUTH:{distance[8]}')
        
                # 臉部黃金比例
                elif drawFortuneTelling == "臉部黃金比例":
                    for idx1,value in enumerate(FIVE_EYE):
                        ID = value

                        lm = faceLms.landmark[ID]
                        # x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                        x, y, z = lm.x*iw, lm.y*ih, lm.z*ic

                        # 起點的 2D int 座標 (給 cv2 用)
                        startAddress2D = int(x), int(top_y)
                        # 終點的 2D int 座標 (給 cv2 用)
                        endAddress2D = int(x), int(bottom_y)

                        self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                        five_eye_x.append(x)

                    if five_eye_x:
                        print(f'x1:{five_eye_x[0]}, x2:{five_eye_x[1]}, x3:{five_eye_x[2]}, x4:{five_eye_x[3]}, x5:{five_eye_x[4]}, x6:{five_eye_x[5]}')

                        total_x = five_eye_x[5] - five_eye_x[0]
                        
                        print(f'total_x:{total_x}')

                        for i in range(len(FIVE_EYE) - 1):
                            x_distance = five_eye_x[i+1] - five_eye_x[i]
                            ratio = x_distance / total_x
                            # print(ratio)
                            five_eye_ratio.append(ratio)

                        print(f'臉部比例為-> {five_eye_ratio[0]}:{five_eye_ratio[1]}:{five_eye_ratio[2]}:{five_eye_ratio[3]}:{five_eye_ratio[4]}')
                        print('------------')

                face = []
                for id, lm in enumerate(faceLms.landmark):   # use enumerate to get index and values 
                    # ih, iw, ic = img.shape
                    x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                    # 畫出臉上每個點的ID
                    if drawID:
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)  # print each ID on face
                    #print(lm)           # print x, y, z of each point
                    #print(id, x, y)
                    #face.append([x,y])  # save x, y without id. 
                    face.append([id,x,y,z])
                faces.append(face)
        return img, faces, distance, sum
    
    # https://google.github.io/mediapipe/solutions/iris.html
    
        
    #美人角
    def angle(self, startPoint, endPoint):
        x1, y1 = startPoint
        x2, y2 = endPoint
        return
        # dx1 = v1[2] - v1[0]
        # dy1 = v1[3] - v1[1]
        # dx2 = v2[2] - v2[0]
        # dy2 = v2[3] - v2[1]
        # angle1 = math.atan2(dy1, dx1)
        # angle1 = int(angle1 * 180/math.pi)
        # # print(angle1)
        # angle2 = math.atan2(dy2, dx2)
        # angle2 = int(angle2 * 180/math.pi)
        # # print(angle2)
        # if angle1*angle2 >= 0:
        #     included_angle = abs(angle1-angle2)
        # else:
        #     included_angle = abs(angle1) + abs(angle2)
        #     if included_angle > 180:
        #         included_angle = 360 - included_angle
        # return included_angle


    # Euclaidean Distance
    def euclaideanDistance(self, startPoint, endPoint):
        x1, y1 = startPoint
        x2, y2 = endPoint
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 )
        return distance

    # Euclaidean Distance
    def euclaideanDistance3D(self, startPoint, endPoint):
        x1, y1, z1 = startPoint
        x2, y2, z2 = endPoint
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 )
        return distance
    
    # draw specific line user defined
    def drawSpecificLine(self, img, startAddress, endAddress, color=DEFAULT_COLOR):
        # address for begin point and destination point
        self.startAddress = startAddress
        self.endAddress= endAddress        
        # using cv2.line to draw line 畫出指定點到點的線
        cv2.line(img, self.startAddress, self.endAddress, color, 3)

        # 就不在這裡算距離了 !!!
        # Euclaidean Distance 計算出距離
        # LineDistance = self.euclaideanDistance(self.startAddress, self.endAddress)
        # LineDistance = self.euclaideanDistance3D(self.startAddress, self.endAddress)
        # return LineDistance

def faceMeshDetection(videoMode=True, filePath="./videos/1-720p.mp4", drawFaceLms=True, drawID=False, drawFortuneTelling="0"):
    #cap = cv2.VideoCapture(0)                         # From camera capture real time 
    #cap = cv2.VideoCapture("../videos/2-1080p.mp4")   # required AV1 hardware decoding
    #cap = cv2.VideoCapture("../videos/2-720pL.mp4")
    #cap = cv2.VideoCapture("../videos/1-720p.mp4")
    # grab global references to the lock variables
    global lock
    if videoMode:
        cap = cv2.VideoCapture(filePath)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=10)
    #faceDetection = True
    while True:
        # success為True/False
        if videoMode:
            success, img = cap.read()
        else:
            #img = cv2.imread('images/03.png')
            img = cv2.imread(filePath)
        # In order to make all video_feed sources working at the same fps, we need to use threading lock. 
        with lock:                  # acquire threading lock, set the output frame, and release threading lock.  
            img, faces, distance, sum = detector.findFaceMesh(img, drawFaceLms, drawID, drawFortuneTelling)      
                    
            # time.time():1970年之後經過的秒数
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            #cv2.imshow("Image", img)
        
            # 傳送至前端
            frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) == ord('q'): 
            break 
    if videoMode:    
        cap.release()   
    cv2.destroyAllWindows()
    return img, faces, distance, sum

if __name__ == "__main__":
    faceMeshDetection()
    


