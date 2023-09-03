import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)
hand = mp.solutions.hands

Hand = hand.Hands(max_num_hands=2)  # Definir la cantidad de manos a detectar}

mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgrgb)
    handsPoint = results.multi_hand_landmarks
    h,w,_ = img.shape
    puntos = []
    if handsPoint:
        for points in handsPoint:
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):
                cx,cy = int(cord.x*w), int(cord.y*h)
                puntos.append((cx,cy))
        dedos = [8,12,16,20]
        contador = 0
        if points:
            if puntos[4][0] < puntos [2][0]:
                contador +=1
            for x in dedos:
                if puntos[x][1] < puntos[x-2][1]:
                    contador += 1


        cv2.rectangle(img,(80,5),(200,110),(255,0,0), -1)
        cv2.putText(img, str(contador),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)
    cv2.imshow('results', img)
    cv2.waitKey(1)