import cv2
import mediapipe as mp

video  = cv2.VideoCapture(0)
hand = mp.solutions.hands

Hand = hand.Hands(max_num_hands=2) # Definir la cantidad de manos a detectar}

mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgrgb)
    handsPoint = results.multi_hand_landmarks
    if handsPoint:
        for points in handsPoint:
            print(points)
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            

    cv2.imshow('results', img)
    cv2.waitKey(1)