import cv2
import time
import numpy as np

sinal = cv2.VideoCapture("sinaleiro.mp4")

kernel = np.ones((5,5),np.uint8)

aux = 0
aux1 = 0
aux2 = 0
aux3 = 0
while (True):

    ret, frame = sinal.read()
    frame_copy = frame.copy()
    
    #BGR
    inf_verde = np.array([[0, 240,0], [30, 30,240],[70, 185,230]])
    sup_verde = np.array([[220, 255,100], [95, 100,255],[90, 255,255],[220, 255,100]])
    
    inf_verme = (30, 30,240)
    sup_verme = (95, 100,255)
    
    inf_amar = (70, 185,230)
    sup_amar = (90, 255,255)

    #Para a cor verde
    mascara_verde = cv2.inRange(frame, inf_verde, sup_verde)
    fechado_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel, iterations = 2)
    dilatar_verde = cv2.dilate(fechado_verde, kernel, iterations = 10)
    fechado_verde = cv2.morphologyEx(dilatar_verde, cv2.MORPH_CLOSE, kernel, iterations = 2)
    afinar_verde = cv2.erode(fechado_verde, kernel, iterations = 10)
    verde, h = cv2.findContours(afinar_verde, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Para a cor vermelho
    mascara_verme = cv2.inRange(frame, inf_verme, sup_verme)
    fechado_verme = cv2.morphologyEx(mascara_verme, cv2.MORPH_CLOSE, kernel, iterations = 2)
    dilatar_verme = cv2.dilate(fechado_verme, kernel, iterations = 10)
    fechado_verme = cv2.morphologyEx(dilatar_verme, cv2.MORPH_CLOSE, kernel, iterations = 2)
    afinar_verme = cv2.erode(fechado_verme, kernel, iterations = 10)
    vermelho, h1 = cv2.findContours(afinar_verme, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Para a cor amarelo
    mascara_amar = cv2.inRange(frame, inf_amar, sup_amar)
    fechado_amar = cv2.morphologyEx(mascara_amar, cv2.MORPH_CLOSE, kernel, iterations = 2)
    dilatar_amar = cv2.dilate(fechado_amar, kernel, iterations = 10)
    fechado_amar = cv2.morphologyEx(dilatar_amar, cv2.MORPH_CLOSE, kernel, iterations = 2)
    afinar_amar = cv2.erode(fechado_amar, kernel, iterations = 10)
    amarelo, h2 = cv2.findContours(afinar_amar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cont in verde:
        M = cv2.moments(cont)
        (x, y, w, h) = cv2.boundingRect(cont)
        if (M['m00'] >= 5000):
            aux1 = 0
            aux2 = 0
            if (aux < 1):
                para_verde = time.time()
                aux = 1
            inicio_verde = time.time()
            tempo_verde = str(round(inicio_verde - para_verde, 5))
            cv2.rectangle(frame_copy, (x-3,y-3), (x+w+2, y+h+2), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame_copy, "verde", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(frame_copy, tempo_verde, (x+200, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

    for cont1 in vermelho:
        M1 = cv2.moments(cont1)
        (x1, y1, w1, h1) = cv2.boundingRect(cont1)
        if (M1['m00'] >= 5000):
            aux = 0
            if (aux1 < 1):
                para_verme = time.time()
                aux1 = 1
                aux2 = -1
            inicio_verme = time.time()
            tempo_verme = str(round(inicio_verme - para_verme, 5))
            cv2.rectangle(frame_copy, (x1-3,y1-3), (x1+w1+2, y1+h1+2), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_copy, "vermelho", (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(frame_copy, tempo_verme, (x1+200, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    
    for cont2 in amarelo:
        M2 = cv2.moments(cont2)
        (x2, y2, w2, h2) = cv2.boundingRect(cont2)
        if (M2['m00'] >= 5000):
            aux = 0
            if (aux2 < 1):
                para_amar = time.time()
                aux2 = aux2 + 1
            inicio_amar = time.time()
            tempo_amar = str(round(inicio_amar - para_amar, 5))
            cv2.rectangle(frame_copy, (x2-3,y2-3), (x2+w2+2, y2+h2+2), (80,180,255), 1, cv2.LINE_AA)
            cv2.putText(frame_copy, "amarelo", (x2+5, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,180,255), 1)
            cv2.putText(frame_copy, tempo_amar, (x2+200, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (80,180,255), 1)

    corte_verde = cv2.bitwise_and(frame, frame, mask = mascara_verde)
    corte_verme = cv2.bitwise_and(frame, frame, mask = mascara_verme)
    corte_amar = cv2.bitwise_and(frame, frame, mask = mascara_amar)
    cv2.imshow("Sinal - Video", frame_copy)
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break
    time.sleep(0.0)
    aux3 = aux3 + 1
    if (aux3 == 1000):
        break
sinal.release()
cv2.destroyAllWindows()

        
        
        
        
        
        