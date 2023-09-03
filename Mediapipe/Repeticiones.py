import math

import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

draw = mp.solutions.drawing_utils
contador = 0
check = True
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec a utilizar
#out = cv2.VideoWriter('Final.mp4', fourcc, 25, (720, 1280))  # crear objeto VideoWriter
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (720, 1280))

while True:
    success, img = video.read()
    videorgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Pose.process(videorgb)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)
    h,w,_ = img.shape
    print(h,w)
    if points:
        pieDy = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].y*h)
        pieDx = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].x*w)
        pieEy = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].y*h)
        pieEx = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].x*w)
        ManoDy = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y*h)
        ManoDx = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x*w)
        ManoEy = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].y*h)
        ManoEx = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].x*w)

        distMD = math.hypot(ManoDx-ManoEx,ManoDy-ManoEy)
        distPe = math.hypot(pieDx - pieEx, pieDy - pieEy)

        if check == True and distMD<=150 and distPe >=150:
            contador+=1
            check = False
        if distMD >150 and distPe < 150:
            check = True

        texto = f'Repeticiones: {contador}'
        #cv2.rectangle(img, (20,20),(600,120),(255,0,0),-1)
        #cv2.putText(img, texto,(50,100), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),5)

    out.write(img)
    cv2.imshow('Resultado', img)
    cv2.waitKey(40)
