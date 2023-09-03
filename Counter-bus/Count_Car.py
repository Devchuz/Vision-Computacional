import cv2
import cvzone
import numpy as np
import os
from ultralytics import YOLO
from sort import *

tracker = Sort(max_age=20)
classNames = ["pessoa", "bicicleta", "carro", "moto", "avião", "ônibus", "trem", "caminhão", "barco",
              "semáforo", "hidrante", "placa de pare", "parquímetro", "banco", "pássaro", "gato",
              "cachorro", "cavalo", "ovelha", "vaca", "elefante", "urso", "zebra", "girafa", "mochila", "guarda-chuva",
              "bolsa", "gravata","mala", "frisbee", "esquis", "snowboard", "bola esportiva", "pipa", "taco de beisebol",
              "luva de beisebol", "skate", "prancha de surf", "raquete de tênis", "garrafa", "copo de vinho", "taça",
              "garfo", "faca", "colher", "tigela", "banana", "maçã", "sanduíche", "laranja", "brócolis",
              "cenoura", "cachorro-quente", "pizza", "rosquinha","bolo", "cadeira", "sofá", "vaso de planta", "cama",
              "mesa de jantar", "banheiro", "monitor de tv", "laptop", "mouse", "controle remoto", "teclado", "telefone celular",
              "microondas", "forno", "torradeira", "pia", "geladeira", "livro", "relógio", "vaso", "tesoura",
              "ursinho de pelúcia", "secador de cabelo", "escova de dentes"
              ]
model = YOLO("yolov8n.pt")
cwd = os.getcwd()
video_path = os.path.join(cwd, 'resource/road.mp4')
video = cv2.VideoCapture(video_path)
mask_path = os.path.join(cwd, 'resource/mask.png')
mask = cv2.imread(mask_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec a utilizar
out = cv2.VideoWriter('.mp4', fourcc, 25, (1100, 700))  # crear objeto VideoWriter
Linea = [110,550,510,550]
Contador = []
while True:
    ret, img = video.read()
    img = cv2.resize(img, (1100, 700))
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    cv2.line(img, (Linea[0], Linea[1]), (Linea[2], Linea[3]), (0, 0, 255), 3)
    for obj in results:
        datos = obj.boxes
        for x in datos:
            x1, y1, x2, y2 = x.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            # cv2.rectangle(img, (x1,y1), (x1+w,y1+h),(255,0,0),3)
            conf = int(x.conf[0] * 100)
            cls = int(x.cls[0])
            nome_class = classNames[cls]

            if conf >= 20 and nome_class == "carro" or nome_class == "caminhão":
                # cvzone.cornerRect(img,(x1,y1,w,h),colorR=(255,0,0))
                # cvzone.putTextRect(img,nome_class,(x1,y1-10),scale=1.5,thickness=2)
                crArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, crArray))
        resultsTracker = tracker.update(detections)

        for results in resultsTracker:
            x1, y1, x2, y2, id = results
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 0))
            # cvzone.putTextRect(img, str(id), (x1, y1 - 10), scale=1.5, thickness=2)
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 2, (255, 0, 0), 2)
            if Linea[0] < cx < Linea[3] and Linea[1] - 15 < cy < Linea[1] + 15:
                if Contador.count(id) == 0:
                    Contador.append(id)
                    cv2.line(img, (Linea[0], Linea[1]), (Linea[2], Linea[3]), (0, 0, 255), 3)

    print(len(Contador))
    cv2.putText(img, str(len(Contador)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    out.write(img)
    cv2.imshow('img',img)


    cv2.waitKey(1)