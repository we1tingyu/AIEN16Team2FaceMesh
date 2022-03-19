
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

import sqlQuery
 

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

    def findFaceMesh(self, img, drawFaceLms=True, drawID=False, drawFortuneTelling="臉部特徵網格圖",takePicture=False, returnTxt=False,returnComment=False):
        # 左右相反，for camera
        #self.imgRGB = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB) # opposite right/left for actual face sync-up detection when face turns left/right.
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        # put faces[]/distance{} here to avoid error"local variable referenced before assignment" 
        faces = []
        distance = []

        ih, iw, ic = img.shape

        FACE_OVAL=facialFeatures.FACE_OVAL
        FACE_OVAL_NEW=facialFeatures.FACE_OVAL_NEW
        READ_FACE=facialFeatures.READ_FACE
        BEAUTY_CORNER=facialFeatures.BEAUTY_CORNER
        #美人角用的座標 
        # beauty_corner_coordinate = []
        startAddressForAngle = []
        endAddressForAngle = []

        # 眼尾和鼻翼
        EYE_AND_NOSE=facialFeatures.EYE_AND_NOSE
        # 眼尾和眉尾
        EYEBROW_AND_EYE=facialFeatures.EYEBROW_AND_EYE

        # 鼻翼
        ALAE_OF_NOSE=facialFeatures.ALAE_OF_NOSE

        # 眼頭
        HEAD_OF_EYE=facialFeatures.HEAD_OF_EYE
        
        # 四角比例
        # 眼尾
        END_OF_EYE=facialFeatures.END_OF_EYE
        # 唇角
        LIP_CORNER=facialFeatures.LIP_CORNER
        # 四角ID
        # SQUARE=facialFeatures.SQUARE
        # leye_ID = SQUARE['leye']
        # reye_ID = SQUARE['reye']
        # SQUARE_POINT=facialFeatures.SQUARE_POINT
        # square_point_y = []

        EDGE=facialFeatures.EDGE
        # 取得四個邊界ID
        # top_ID = EDGE['top']
        # bottom_ID = EDGE['bottom']
        # left_ID = EDGE['left']
        # right_ID = EDGE['right']
        top_ID = EDGE[0]
        bottom_ID = EDGE[1]
        left_ID = EDGE[2]
        right_ID = EDGE[3]

        THREE_COURT=facialFeatures.THREE_COURT


        three_court_y = []
        three_court_ratio = []

        FIVE_EYE=facialFeatures.FIVE_EYE
        five_eye_x = []
        five_eye_ratio = []

        ratio_diff = []
        score=0
        sum=0  
        printTxt = ""
        printComment = ""
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

                # 取得四個邊界的 x 和 y
                top_y = faceLms.landmark[top_ID].y*ih
                bottom_y = faceLms.landmark[bottom_ID].y*ih
                # left_x 和 right_x 初始值
                left_x = faceLms.landmark[left_ID].x*iw
                right_x = faceLms.landmark[right_ID].x*iw

                # 確認 left_x, left_ID 和 right_x, 
                for idx1,value in enumerate(FACE_OVAL_NEW):
                    ID = value

                    lm_ID = faceLms.landmark[ID]
                    x1, y1, z1 = lm_ID.x*iw, lm_ID.y*ih, lm_ID.z*ic

                    if x1 < left_x:
                        left_x = x1
                        left_ID = ID
                    elif x1 > right_x:
                        right_x = x1
                        right_ID = ID

                print(f'left_x:{left_x}, left_ID:{left_ID}, right_x:{right_x}, right_ID:{right_ID}, ')
                
                # 計算髮際線的座標, 用原本臉長 * 1 / 0.87 倍當做實際臉長
                top_y = bottom_y - (bottom_y - top_y) * (1 / 0.87)

                # 算出臉的長(total_y)寬(total_x)
                total_y = bottom_y - top_y
                total_x = right_x - left_x
                
                if returnComment:
                    printComment = "<h2>評論:</h2>"

                # 臉部特徵網格圖, 什麼都不畫
                if drawFortuneTelling == "臉部特徵網格圖":
                    printTxt += "什麼都不告訴你"
                    if returnComment:
                        printComment += "什麼都不告訴你"
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
                            # draw specific line user defined (只畫線, 不算距離)
                            self.drawSpecificLine(img, startAddress2D, endAddress2D)

                            # 起點的 3D float 座標
                            startAddress3D = x1, y1, z1
                            # 終點的 3D float 座標
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
                        
                    # print(f'臉的周長:{sum:.2f}')
                    # print('------------')

                    printTxt += f'臉的周長:{sum:.2f}<br>'

                    if returnComment:
                        printComment += "還是什麼都不告訴你唷"

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
                            # draw specific line user defined (只畫線, 不算距離)
                            self.drawSpecificLine(img, startAddress2D, endAddress2D, BLACK)
                            
                            # 起點的 3D float 座標
                            startAddress3D = x1, y1, z1
                            # 終點的 3D float 座標
                            endAddress3D = x2, y2, z2

                            # Euclaidean Distance 計算出距離
                            # 計算 2D 距離
                            # lineDistance = self.euclaideanDistance(startAddress2D, endAddress2D)
                            # 計算 3D 距離
                            lineDistance = self.euclaideanDistance3D(startAddress3D, endAddress3D)
                            sum+=lineDistance

                        # append into distance[]                           
                        distance.append(sum)                     
                    if distance:
                        # print(f'RIGHT_EYEBROW:{distance[0]:.2f}, LEFT_EYEBROW:{distance[1]:.2f}, RIGHT_EYE:{distance[2]:.2f}, LEFT_EYE{distance[3]:.2f}, NOSE_LENGTH:{distance[4]:.2f}, NOSE_WIDTH:{distance[5]:.2f}, FOREHEAD:{distance[6]:.2f}, PHILTRUM:{distance[7]:.2f}, MOUTH:{distance[8]:.2f}')
                        # print('------------')

                        printTxt += f'RIGHT_EYEBROW:{distance[0]:.2f}, LEFT_EYEBROW:{distance[1]:.2f}, RIGHT_EYE:{distance[2]:.2f}, LEFT_EYE{distance[3]:.2f}, NOSE_LENGTH:{distance[4]:.2f}, NOSE_WIDTH:{distance[5]:.2f}, FOREHEAD:{distance[6]:.2f}, PHILTRUM:{distance[7]:.2f}, MOUTH:{distance[8]:.2f}<br>'

                        if returnComment:
                            printComment += "就是什麼都不告訴你啦"

                # 三庭
                elif drawFortuneTelling == "三庭":
                    for idx1,value in enumerate(THREE_COURT):
                        ID = value

                        lm = faceLms.landmark[ID]
                        # x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                        x, y, z = lm.x*iw, lm.y*ih, lm.z*ic

                        # 修正髮際線的 y 座標
                        if idx1 == 0:
                            y = top_y
                        
                        # 起點的 2D int 座標 (給 cv2 用)
                        startAddress2D = int(left_x), int(y)
                        # 終點的 2D int 座標 (給 cv2 用)
                        endAddress2D = int(right_x), int(y)
                        # 畫線
                        self.drawSpecificLine(img, startAddress2D, endAddress2D, RED)

                        three_court_y.append(y)

                    if three_court_y:
                        # print(f'由上到下的 y 座標分別是 y1:{three_court_y[0]:.2f}, y2:{three_court_y[1]:.2f}, y3:{three_court_y[2]:.2f}, y4:{three_court_y[3]:.2f}')

                        printTxt += f'由上到下的 y 座標分別是 y1:{three_court_y[0]:.2f}, y2:{three_court_y[1]:.2f}, y3:{three_court_y[2]:.2f}, y4:{three_court_y[3]:.2f}<br>'

                        # print(f'上到下的 y 距離:{total_y:.2f}')

                        printTxt += f'上到下的 y 距離:{total_y:.2f}<br>'

                        for i in range(len(THREE_COURT) - 1):
                            y_distance = three_court_y[i+1] - three_court_y[i]
                            ratio = (y_distance / total_y) * 3
                            # print(ratio)
                            three_court_ratio.append(ratio)
                            ratio_diff.append(1-ratio)
                            score += 1-(abs(1-ratio))
                        score = score * 100 / 3

                        # print(f'三庭(上到下)比例為-> {three_court_ratio[0]:.2f}:{three_court_ratio[1]:.2f}:{three_court_ratio[2]:.2f}')
                        # print('------------')

                        printTxt += f'三庭(上到下)比例為-> {three_court_ratio[0]:.2f}:{three_court_ratio[1]:.2f}:{three_court_ratio[2]:.2f}<br>'
                        printTxt += f'三庭的完美比例是-> 1:1:1<br>'
                        printTxt += f'您的落差為-> {ratio_diff[0]:.2f}:{ratio_diff[1]:.2f}:{ratio_diff[2]:.2f}<hr>'
                        printTxt += f'您獲得的分數為-> {score:.2f}分'

                        if returnComment:
                            # 取得 comment
                            button_name = drawFortuneTelling
                            comment_level = int(score / 20)
                            printComment += self.getComment(button_name, comment_level)
                
                # 五眼
                elif drawFortuneTelling == "五眼":
                    for idx1,value in enumerate(FIVE_EYE):
                        ID = value

                        lm = faceLms.landmark[ID]
                        # x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                        x, y, z = lm.x*iw, lm.y*ih, lm.z*ic

                        # 起點的 2D int 座標 (給 cv2 用)
                        startAddress2D = int(x), int(top_y)
                        # 終點的 2D int 座標 (給 cv2 用)
                        endAddress2D = int(x), int(bottom_y)
                        # 畫線
                        self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                        five_eye_x.append(x)
                        

                    if five_eye_x:
                        # print(f'由左到右的 x 座標分別是 x1:{five_eye_x[0]:.2f}, x2:{five_eye_x[1]:.2f}, x3:{five_eye_x[2]:.2f}, x4:{five_eye_x[3]:.2f}, x5:{five_eye_x[4]:.2f}, x6:{five_eye_x[5]:.2f}')

                        printTxt += f'由左到右的 x 座標分別是 x1:{five_eye_x[0]:.2f}, x2:{five_eye_x[1]:.2f}, x3:{five_eye_x[2]:.2f}, x4:{five_eye_x[3]:.2f}, x5:{five_eye_x[4]:.2f}, x6:{five_eye_x[5]:.2f}<br>'

                        # print(f'左到右的 x 距離:{total_x:.2f}')

                        printTxt += f'左到右的 x 距離:{total_x:.2f}<br>'

                        for i in range(len(FIVE_EYE) - 1):
                            x_distance = five_eye_x[i+1] - five_eye_x[i]
                            ratio = (x_distance / total_x) * 5
                            # print(ratio)
                            five_eye_ratio.append(ratio)
                            ratio_diff.append(1-ratio)
                            score += 1-(abs(1-ratio))
                        score = score * 100 / 5

                        # print(f'五眼(左到右)比例為-> {five_eye_ratio[0]:.2f}:{five_eye_ratio[1]:.2f}:{five_eye_ratio[2]:.2f}:{five_eye_ratio[3]:.2f}:{five_eye_ratio[4]:.2f}')
                        # print('------------')

                        printTxt += f'五眼(左到右)比例為-> {five_eye_ratio[0]:.2f}:{five_eye_ratio[1]:.2f}:{five_eye_ratio[2]:.2f}:{five_eye_ratio[3]:.2f}:{five_eye_ratio[4]:.2f}<hr>'
                        printTxt += f'五眼的完美比例是-> 1:1:1:1:1<br>'
                        printTxt += f'您的落差為-> {ratio_diff[0]:.2f}:{ratio_diff[1]:.2f}:{ratio_diff[2]:.2f}:{ratio_diff[3]:.2f}:{ratio_diff[4]:.2f}<hr>'
                        printTxt += f'您獲得的分數為-> {score:.2f}分'
                        
                        if returnComment:
                            # 取得 comment
                            button_name = drawFortuneTelling
                            comment_level = int(score / 20)
                            printComment += self.getComment(button_name, comment_level)

                #美人角
                elif drawFortuneTelling == "美人角":
                    for idx1,ff in enumerate(BEAUTY_CORNER):
                        for idx2,value in enumerate(BEAUTY_CORNER[idx1]):
                            startID, endID = value
                           
                            lm_start = faceLms.landmark[startID]
                            x1, y1, z1 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                            # 起點的 2D int 座標 (給 cv2 用)
                            startAddress2D = int(x1), int(y1)
                            lm_end = faceLms.landmark[endID]
                            x2, y2, z2 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic
                            # 終點的 2D int 座標 (給 cv2 用)
                            endAddress2D = int(x2), int(y2)
                            # draw specific line user defined (只畫線, 不算距離)
                            self.drawSpecificLine(img, startAddress2D, endAddress2D, BLACK)
                            
                            # 起點的 3D float 座標
                            startAddressForAngle.append([x1, y1])
                            # 終點的 3D float 座標
                            endAddressForAngle.append([x2, y2])

                    # 計算夾角
                    ang1 = self.angle(startAddressForAngle[0], endAddressForAngle[0], startAddressForAngle[1], endAddressForAngle[1])
                    # print(f"美人角角度是{ang1}°")

                    score = (1 - (abs(ang1-45) / 45)) * 100

                    printTxt += f"美人角角度是-> {ang1}°<br>"
                    printTxt += f'美人角的完美角度是-> 45°<br>'
                    printTxt += f'您的落差為-> {abs(ang1-45)}°<hr>'
                    printTxt += f'您獲得的分數為-> {score:.2f}分'

                    if returnComment:
                        # 取得 comment
                        button_name = drawFortuneTelling
                        comment_level = int(score / 20)
                        printComment += self.getComment(button_name, comment_level)
                
                # 臉部黃金比例
                elif drawFortuneTelling == "臉部黃金比例":
                    # 起點的 2D int 座標 (給 cv2 用)
                    startAddress2D = int(left_x), int(top_y)
                    # 終點的 2D int 座標 (給 cv2 用)
                    endAddress2D = int(right_x), int(top_y)
                    # 畫線
                    self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                    # 起點的 2D int 座標 (給 cv2 用)
                    startAddress2D = int(right_x), int(top_y)
                    # 終點的 2D int 座標 (給 cv2 用)
                    endAddress2D = int(right_x), int(bottom_y)
                    # 畫線
                    self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                    face_ratio = total_y / total_x

                    # 鑑別值
                    k=1
                    # 調整黃金比例
                    GR = 1.618
                    
                    score = (1-(k*abs(face_ratio-GR)/GR))*100
                    
                    printTxt += f'臉部比例為-> 1:{face_ratio:.3f}<hr>'                    
                    printTxt += f'您獲得的分數為-> {score:.2f}分<br>'

                    if returnComment:
                        # 取得 comment
                        button_name = drawFortuneTelling
                        comment_level = int(score / 20)
                        printComment += self.getComment(button_name, comment_level)

                # 臉部四角形比例
                elif drawFortuneTelling == "臉部四角形比例":
                    # 眼尾的線
                    startID, endID = END_OF_EYE

                    lm_start = faceLms.landmark[startID]
                    x1, y1, z1 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                    # # 起點的 2D int 座標 (給 cv2 用)
                    # startAddress2D = int(x1), int(y1)
                    lm_end = faceLms.landmark[endID]
                    x2, y2, z2 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic

                    y_average_top = (y1 + y2) / 2
                    # 起點的 2D int 座標 (給 cv2 用)
                    startAddress2D = int(x1), int(y_average_top)
                    # 終點的 2D int 座標 (給 cv2 用)
                    endAddress2D = int(x2), int(y_average_top)
                    # draw specific line user defined (只畫線, 不算距離)
                    self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                    # 唇角的線
                    startID, endID = LIP_CORNER

                    lm_start = faceLms.landmark[startID]
                    x3, y3, z3 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                    # # 起點的 2D int 座標 (給 cv2 用)
                    # startAddress2D = int(x1), int(y1)
                    lm_end = faceLms.landmark[endID]
                    x4, y4, z4 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic

                    y_average_bottom = (y3 + y4) / 2
                    # 起點的 2D int 座標 (給 cv2 用)
                    startAddress2D = int(x1), int(y_average_bottom)
                    # 終點的 2D int 座標 (給 cv2 用)
                    endAddress2D = int(x2), int(y_average_bottom)
                    # draw specific line user defined (只畫線, 不算距離)
                    self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                    # 畫左邊的線
                    # 起點的 2D int 座標 (給 cv2 用)
                    startAddress2D = int(x1), int(y_average_top)
                    # 終點的 2D int 座標 (給 cv2 用)
                    endAddress2D = int(x1), int(y_average_bottom)
                    # draw specific line user defined (只畫線, 不算距離)
                    self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                    # 畫右邊的線
                    # 起點的 2D int 座標 (給 cv2 用)
                    startAddress2D = int(x2), int(y_average_top)
                    # 終點的 2D int 座標 (給 cv2 用)
                    endAddress2D = int(x2), int(y_average_bottom)
                    # draw specific line user defined (只畫線, 不算距離)
                    self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                    four_square_ratio = (x2 - x1) / (y_average_bottom - y_average_top)
                    # print(f'臉部四角形長寬分別為: {(x2 - x1):.2f}, {(y_average_bottom - y_average_top):.2f}')
                    # print(f'臉部四角形比例為: {four_square_ratio:.2f}')
                    # print('------------')

                    printTxt += f'臉部四角形長寬分別為: {(x2 - x1):.2f}, {(y_average_bottom - y_average_top):.2f}<br>'
                    printTxt += f'臉部四角形比例為: {four_square_ratio:.2f}<hr>'
                    
                    # 鑑別值
                    k=1
                    # 完美鼻寬比值
                    PerfectFC = 1.4 
                    score = (1-(k*abs(four_square_ratio-PerfectFC)/PerfectFC))*100          
                    
                    printTxt += f'您獲得的分數為-> {score:.2f}分<br>'

                    if returnComment:
                        # 取得 comment
                        button_name = drawFortuneTelling
                        comment_level = int(score / 20)
                        printComment += self.getComment(button_name, comment_level)

                # 眉尾、眼尾和鼻翼連成一線
                elif drawFortuneTelling == "眉尾、眼尾和鼻翼連成一線":
                    # 眉尾和眼尾
                    for idx1,ff in enumerate(EYEBROW_AND_EYE):
                        for idx2,value in enumerate(EYEBROW_AND_EYE[idx1]):
                            startID, endID = value
                           
                            lm_start = faceLms.landmark[startID]
                            x1, y1, z1 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                            # 起點的 2D int 座標 (給 cv2 用)
                            startAddress2D = int(x1), int(y1)
                            lm_end = faceLms.landmark[endID]
                            x2, y2, z2 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic
                            # 終點的 2D int 座標 (給 cv2 用)
                            endAddress2D = int(x2), int(y2)
                            # draw specific line user defined (只畫線, 不算距離)
                            self.drawSpecificLine(img, startAddress2D, endAddress2D, RED)
                            
                            # 起點的 3D float 座標
                            startAddressForAngle.append([x1, y1])
                            # 終點的 3D float 座標
                            endAddressForAngle.append([x2, y2])

                    # 計算眼尾和眉尾夾角
                    ang1 = self.angle(startAddressForAngle[0], endAddressForAngle[0], startAddressForAngle[1], endAddressForAngle[1])
                    # print(f"眼尾和眉尾夾角角度是{ang1}°")
                    printTxt += f"眼尾和眉尾夾角角度是{ang1}°<br>"

                    # 眼尾和鼻翼
                    for idx1,ff in enumerate(EYE_AND_NOSE):
                        for idx2,value in enumerate(EYE_AND_NOSE[idx1]):
                            startID, endID = value
                           
                            lm_start = faceLms.landmark[startID]
                            x1, y1, z1 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                            # 起點的 2D int 座標 (給 cv2 用)
                            startAddress2D = int(x1), int(y1)
                            lm_end = faceLms.landmark[endID]
                            x2, y2, z2 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic
                            # 終點的 2D int 座標 (給 cv2 用)
                            endAddress2D = int(x2), int(y2)
                            # draw specific line user defined (只畫線, 不算距離)
                            self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)
                            
                            # 起點的 3D float 座標
                            startAddressForAngle.append([x1, y1])
                            # 終點的 3D float 座標
                            endAddressForAngle.append([x2, y2])

                    # 計算眼尾和鼻翼夾角
                    ang2 = self.angle(startAddressForAngle[2], endAddressForAngle[2], startAddressForAngle[3], endAddressForAngle[3])
                    # print(f"眼尾和鼻翼夾角角度是{ang2}°")

                    printTxt += f"眼尾和鼻翼夾角角度是{ang2}°<br>"

                    # 計算(左眉-左眼)和(左眼-左鼻)夾角, (正數)順時鐘旋轉表示眉毛較短
                    ang3 = self.angle(startAddressForAngle[0], endAddressForAngle[0], startAddressForAngle[2], endAddressForAngle[2], ignore_clockwise_direction=False)
                    # print(f"左眉尾、左眼尾和左鼻翼夾角角度是{180-ang3}° (若為 180° 表示連成一直線, 大於 180° 表示眉毛較長, 小於 180° 表示眉毛較短)")

                    printTxt += f"左眉尾、左眼尾和左鼻翼夾角角度是{180-ang3}° (若為 180° 表示連成一直線, 大於 180° 表示眉毛較長, 小於 180° 表示眉毛較短)<br>"

                    # 計算(右眉-右眼)和(右眼-右鼻)夾角, (負數)逆時鐘旋轉表示眉毛較短, 故取負數
                    ang4 = -self.angle(startAddressForAngle[1], endAddressForAngle[1], startAddressForAngle[3], endAddressForAngle[3], ignore_clockwise_direction=False)
                    # print(f"右眉尾、右眼尾和右鼻翼夾角角度是{180-ang4}° (若為 180° 表示連成一直線, 大於 180° 表示眉毛較長, 小於 180° 表示眉毛較短)")
                    # print('------------')

                    printTxt += f"右眉尾、右眼尾和右鼻翼夾角角度是{180-ang4}° (若為 180° 表示連成一直線, 大於 180° 表示眉毛較長, 小於 180° 表示眉毛較短)<hr>"
                    
                    ThreePointsLineL= 180-ang3
                    ThreePointsLineR= 180-ang4
                    
                    # 鑑別值
                    k=5
                    # 完美鼻寬比值
                    PerfectTPL = 180 
                    
                    scoreL = (1-(k*abs(ThreePointsLineL-PerfectTPL)/PerfectTPL))*100 
                    scoreR = (1-(k*abs(ThreePointsLineR-PerfectTPL)/PerfectTPL))*100          
                    score = (scoreL+scoreR)/2
                    printTxt += f'您獲得的分數為-> {score:.2f}分<br>'

                    if returnComment:
                        # 取得 comment
                        button_name = drawFortuneTelling
                        comment_level = int(score / 20)
                        printComment += self.getComment(button_name, comment_level)
                    
                # 鼻子大小
                elif drawFortuneTelling == "鼻子大小":
                    # 開始計算鼻翼
                    startID, endID = ALAE_OF_NOSE
                    
                    lm_start = faceLms.landmark[startID]
                    x1, y1, z1 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                    # 起點的 2D int 座標 (給 cv2 用)
                    startAddress2D = int(x1), int(y1)
                    lm_end = faceLms.landmark[endID]
                    x2, y2, z2 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic
                    # 終點的 2D int 座標 (給 cv2 用)
                    endAddress2D = int(x2), int(y2)
                    # draw specific line user defined (只畫線, 不算距離)
                    self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                    alae_of_nose_ratio = (x2 - x1) / total_x
                    # print(f'鼻子寬度佔臉部寬度比例為: {alae_of_nose_ratio:.2f}')
                    
                    printTxt += f'鼻子寬度佔臉部寬度比例為: {alae_of_nose_ratio:.2f}<br>'

                    # 開始計算眼頭
                    startID, endID = HEAD_OF_EYE
                    
                    lm_start = faceLms.landmark[startID]
                    x1, y1, z1 = lm_start.x*iw, lm_start.y*ih, lm_start.z*ic
                    # 起點的 2D int 座標 (給 cv2 用)
                    startAddress2D = int(x1), int(y1)
                    lm_end = faceLms.landmark[endID]
                    x2, y2, z2 = lm_end.x*iw, lm_end.y*ih, lm_end.z*ic
                    # 終點的 2D int 座標 (給 cv2 用)
                    endAddress2D = int(x2), int(y2)
                    # draw specific line user defined (只畫線, 不算距離)
                    self.drawSpecificLine(img, startAddress2D, endAddress2D, GREEN)

                    head_of_eye_ratio = (x2 - x1) / total_x
                    # print(f'眼頭寬度佔臉部寬度比例為: {head_of_eye_ratio:.2f}')
                    # print(f'兩者比例為: {alae_of_nose_ratio / head_of_eye_ratio:.2f}')
                    # print('------------')
                    NoseWide = alae_of_nose_ratio / head_of_eye_ratio
                    
                    # 鑑別值
                    k=1.7
                    # 完美鼻寬比值
                    PerfectNW = 1 
                    
                    scoreGirl = (1-(k*abs(NoseWide-PerfectNW)/PerfectNW))*100
                    scoreBoy = (1-(k*abs(NoseWide-PerfectNW*1.2)/PerfectNW*1.2))*100
                    
                    
                    printTxt += f'眼頭寬度佔臉部寬度比例為: {head_of_eye_ratio:.2f}<br>'
                    printTxt += f'兩者比例為: {NoseWide:.2f}<hr>' 
                    printTxt += f'若為男性您獲得的分數為-> {scoreBoy:.2f}分<br>'                    
                    printTxt += f'若為女性您獲得的分數為-> {scoreGirl:.2f}分<br>'

                    if returnComment:
                        # 取得 comment
                        button_name = drawFortuneTelling
                        comment_level = int(scoreBoy / 20)
                        printComment += "若為男性-> " + self.getComment(button_name, comment_level) + "<br>"
                        button_name = drawFortuneTelling
                        comment_level = int(scoreGirl / 20)
                        printComment += "若為女性-> " + self.getComment(button_name, comment_level) + "<br>"

                if returnTxt:
                    return printTxt, printComment

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
    
    # sql 查詢 comment
    def getComment(self, button_name, comment_level):
        # sql = f"select A.comment from comment_table A join button_table B on A.button_id = B.button_id where A.comment_level = {int(score / 20)} and B.button_name = '三庭';"
        # sqlComment = sqlQuery.sqlQuery(sql)
        sqlComment = sqlQuery.sqlQueryComment(button_name, comment_level)
        # print(sqlComment[0]['comment'])
        return sqlComment[0]['comment']
    
    # 給四個座標點(起始點1, 終止點1, 起始點2, 終止點2)計算夾角
    def angle(self, startAddress1, endAddress1, startAddress2, endAddress2, ignore_clockwise_direction=True):
        dx1 = endAddress1[0] - startAddress1[0]
        dy1 = endAddress1[1] - startAddress1[1]
        dx2 = endAddress2[0] - startAddress2[0]
        dy2 = endAddress2[1] - startAddress2[1]

        # 向量1和x軸的夾角 θ 為(-π < θ ≤ π)之間
        angle1 = math.atan2(dy1, dx1)
        # 向量1和x軸的夾角 θ 為(-180° < θ ≤ 180°)之間
        angle1 = int(angle1 * 180/math.pi)
        # print(angle1)
        # 向量2和x軸的夾角 θ 為(-π < θ ≤ π)之間
        angle2 = math.atan2(dy2, dx2)
        # 向量2和x軸的夾角 θ 為(-180° < θ ≤ 180°)之間
        angle2 = int(angle2 * 180/math.pi)
        # print(angle2)
        if angle1*angle2 >= 0:
            included_angle = abs(angle1-angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        # ignore_clockwise_direction 為 False, 表示向量1到向量2順時鐘旋轉的角度
        if ignore_clockwise_direction == False and abs(angle1) < abs(angle2):
            # included_angle 為正數表示向量1到向量2順時鐘旋轉的角度, 負數表示向量1到向量2逆時鐘旋轉的角度
            included_angle = -included_angle
        return included_angle

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

def faceMeshDetection(videoMode=True, filePath="./videos/1-720p.mp4", drawFaceLms=True, drawID=False, drawFortuneTelling="臉部特徵網格圖"):
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
                    
            # 若為影片則加上 FPS
            # if videoMode:
            #     # time.time():1970年之後經過的秒数
            #     cTime = time.time()
            #     fps = 1/(cTime-pTime)
            #     pTime = cTime
            #     cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            #     #cv2.imshow("Image", img)
        
            # 傳送至前端
            frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # 若為圖片則直接 break 出 while 迴圈
        if not videoMode:
            # cv2.imencode('.jpg',
            # img)[1].tofile("./app/static/images/" +drawFortuneTelling+ "YAYAYA.jpg") 
            # print("success to save:" + drawFortuneTelling + "YAYAYA.jpg")
            # print("-------------------------")
            break
            
        if cv2.waitKey(1) == ord('q'): 
            break 
        
    if videoMode:    
        cap.release()   
    cv2.destroyAllWindows()
    return img, faces, distance, sum

if __name__ == "__main__":
    faceMeshDetection()
    


