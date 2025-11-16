import cv2
import time
import numpy as np

sinal = cv2.VideoCapture("sinaleiro.mp4")
kernel = np.ones((5,5),np.uint8)

j = 0
aux1 = 0
temp = 0
temp1 = 0
aux =            [1,1,1]
inicio =         [0,0,0]
tempo =          [0,0,0]
area =          [0,0,0]
area1 = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
while (True):

    ret, frame = sinal.read()
    frame_copy = frame.copy()
    #BGR
    inf  = np.array([  (0, 240,0),       (80, 30,240),   (70, 175,240)])
    sup  = np.array([  (220, 255,100),   (110, 90,255),  (90, 255,255)])
    nome = np.array([  ("verde"),        ("vermelho"),   ("amarelo")])
    sup1 =            ((0, 255, 0),      (0, 0,255),     (80,180,255)) 
    for i in range(3):
        
        mascara = cv2.inRange(frame, inf[i], sup[i] )
        fechado = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations = 2)
        dilatar = cv2.dilate(fechado, kernel, iterations = 10)
        fechado = cv2.morphologyEx(dilatar, cv2.MORPH_CLOSE, kernel, iterations = 2)
        c, h = cv2.findContours(fechado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        parcial = 0
        area[i] = (area[i] + area1[i][j])/(12)        
        for cont in c:
            M = cv2.moments(cont)
            (x, y, w, h) = cv2.boundingRect(cont)
            if (parcial <= M['m00']):
                parcial = M['m00']
            area1[i][j] = parcial
            if (area[i] <= 1000):
                aux[i] = 0
            if (M['m00'] >= 13000):
                if (aux[i] == 0):
                    aux[i] = 1
                    inicio[i] = time.time() 
                para = time.time()
                tempo[i] = str(round(para - inicio[i], 5))
                for k in tempo:
                    temp = (temp + float(k))
                temp = temp/len(tempo)
                cv2.rectangle(frame_copy, (x-3,y-3), (x+w+2, y+h+2), sup1[i], 1, cv2.LINE_AA)
                cv2.putText(frame_copy, nome[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sup1[i], 1)
                cv2.putText(frame_copy, tempo[i], (x+180, y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, sup1[i], 3)
    corte = cv2.bitwise_and(frame, frame, mask = mascara)
    cv2.imshow("Sinal - Video", frame_copy)
    if (cv2.waitKey(20) & 0xFF == ord("q")):
        break
    time.sleep(0.0)
    aux1 = aux1 + 1
    if (aux1 == 1000):
        break
j = j + 1        
if (j == 13):
    j = 0
sinal.release()
cv2.destroyAllWindows()

        
        
        
        
        
        