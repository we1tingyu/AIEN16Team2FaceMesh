#
# AIEN16 第二組 臉部偵測Sample code --> code reference from google mediapipe website.
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

lock = threading.Lock()

GREEN = (0, 255, 0)
FACEMESH_FORTUNE_TELLING = {"Left eyebrow":[46, 55], "Right eyebrow":[285, 276], "Nose bridge":[9, 164],
                             "Mouth width":[57, 287], "Nose width":[36, 266]}
#
#  facial research: https://www.researchgate.net/figure/Linear-measures-11-to-17-11-Upper-facial-width-Zid-and-Zie-12-Lower-facial-width_fig4_262762692
#FACEMESH_FORTUNE_TELLING = frozenset([(46, 55), (285, 276), (9, 164), (57, 287), (36, 266)])
#
class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=10, minDetectionCon=False, minTrackCon=0.5):
        
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)  # there are 468 points that's a lot, we need to know the actual position of each point. 

    def findFaceMesh(self, img, drawFaceLms=True, drawID=False, drawFortuneTelling=True):
        #self.imgRGB = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB) # opposite right/left for actual face sync-up detection when face turns left/right.
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        # put faces/distance here to avoid error"local variable referenced before assignment" 
        faces = []
        distance = {}
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:                
                if drawFaceLms:
                    # image=img, landmark_list=faceLms, connections=self.mpFaceMesh.FACEMESH_TESSELATION, 
                    # landmark_drawing_spec=self.drawSpec,
                    # connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_tesselation_style().
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                                self.drawSpec, self.mpDrawingStyles.get_default_face_mesh_tesselation_style())
                    # image=img, landmark_list=faceLms, connections=self.mpFaceMesh.FACEMESH_CONTOURS, 
                    # landmark_drawing_spec=self.drawSpec,
                    # connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_contours_style().
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                                self.drawSpec, self.mpDrawingStyles.get_default_face_mesh_contours_style())
                if drawFortuneTelling:
                    # draw specific IDs for fortune telling
                    for key, value in FACEMESH_FORTUNE_TELLING.items():
                        startID, endID = value
                        ih, iw, ic = img.shape
                        # Transform startID to address(x,y)
                        lm = faceLms.landmark[startID]
                        x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                        startAddress = x, y
                        # Transform endID to address(x,y)
                        lm = faceLms.landmark[endID]
                        x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                        endAddress = x, y
                        # draw ID on image
                        cv2.putText(img, str(startID), startAddress, cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)
                        cv2.putText(img, str(endID), endAddress, cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)                        
                        # calculate distance 
                        lineDistance = self.drawSpecificLine(img, startAddress, endAddress)
                        # append key, value into distance{}.
                        distance[key] = int(lineDistance)
                #                         
                # for lm in faceLms.landmark:
                #     print(lm)     # print x, y, z of each point
                face = []
                for id, lm in enumerate(faceLms.landmark):   # use enumerate to get index and values 
                    ih, iw, ic = img.shape
                    x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                    if drawID:
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)  # print each ID on face
                    #print(lm)           # print x, y, z of each point
                    #print(id, x, y)
                    #face.append([x,y])  # save x, y without id. 
                    face.append([id,x,y,z])
                faces.append(face)
        return img, faces, distance
    #
    # https://google.github.io/mediapipe/solutions/iris.html
    # Euclaidean Distance
    def euclaideanDistance(self, startPoint, endPoint):
        x, y = startPoint
        x1, y1 = endPoint
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance
    
    # draw specific line user defined
    def drawSpecificLine(self, img, startAddress, endAddress):
        # address for begin point and destination point
        self.startAddress = startAddress
        self.endAddress= endAddress        
        # using cv2.line to draw line
        cv2.line(img, self.startAddress, self.endAddress, GREEN, 1)        
        # Euclaidean Distance
        LineDistance = self.euclaideanDistance(self.startAddress, self.endAddress)
        return LineDistance
#
def faceMeshDetection(videoMode=True, filePath="../videos/1-720p.mp4", drawFaceLms=True, drawID=False, drawFortuneTelling=True):
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
        if videoMode:
            success, img = cap.read()
        else:
            #img = cv2.imread('../images/03.png')
            img = cv2.imread(filePath)
        # In order to make all video_feed sources working at the same fps, we need to use threading lock. 
        with lock:                        # acquire threading lock, set the output frame, and release threading lock.
            img, faces, distance = detector.findFaceMesh(img, drawFaceLms, drawID, drawFortuneTelling)
            # if len(faces) != 0:
            #     #print(len(faces))      # see how many faces have been detected.
            #     #print(len(faces[0]))   # see how many points have been detected at the first detected face.
            #     if drawFortuneTelling:
            #         print(distance)     # see all distances defined by fortune telling.
            #     else:
            #         print(faces[0])     # see all points at 1st face are listed.  
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            #cv2.imshow("Image", img)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'): 
            break
    if videoMode:    
        cap.release()
    cv2.destroyAllWindows()
    return img, faces, distance

if __name__ == "__main__":
    faceMeshDetection()
