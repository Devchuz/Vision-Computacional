import cv2
import mediapipe as mp


video = cv2.VideoCapture(0)

mpFaceMesh = mp.solutions.face_mesh

faceMesh = mpFaceMesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    img = cv2.resize(img,(1000,720))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    results = faceMesh.process(imgRGB)
    if results:
        for face in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(img,face,mpFaceMesh.FACEMESH_FACE_OVAL)
            # Dibujar todos los puntos de la cara

            dilx, dily = int(face.landmark[159].x*w), int(face.landmark[159].y*h)
            print(dilx,dily)


    cv2.imshow('Img', img)
    cv2.waitKey(1)