import cv2
import time
import numpy as np

sinal = cv2.VideoCapture("project.mp4")

kernel = np.ones((5,5),np.uint8)

aux = 0
aux1 = 0
aux2 = 0
aux3 = 0
auxB = 0
auxR = 0
auxY = 0
azul = 0
vermelho  = 0
amarelo = 0
cor = 0
inicio = np.array([0,0,0])
verm = np.array([0,0,0,0,0])
pos = 0
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
        afinar = cv2.erode(fechado, kernel, iterations = 10)
        c, h = cv2.findContours(afinar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
        for cont in c:

            M = cv2.moments(cont)
            (x, y, w, h) = cv2.boundingRect(cont)
    
            if (i == 0):
                azul = M['m00'] 
            if (i == 1):
                verm[pos] = M['m00'] 
                pos  = pos + 1
                if (pos == 4):
                    pos = 0
            for j in verm:
                vermelho = vermelho + j
            vermelho = vermelho/(len(verm)-1)
            print("Média vermelho: {}\n".format(vermelho))
            if (i == 2):
                amarelo = M['m00'] 
            if (azul < 4000):
                auxB = 0
            if (vermelho < 300):
                auxR = 0
            if (amarelo < 20):
                auxY = 0
                
            print("azul: {} + verm: {} + amar: {} \n".format(azul, vermelho, amarelo))
            time.sleep(0.0)
            if (M['m00'] >= 5100):
                
                if (azul >= 5100 and auxB == 0):
                    auxB = 1
                    inicio[i] = time.time()                
                if (vermelho >= 5100 and auxR == 0):
                    auxR = 1
                    inicio[i] = time.time()
                if (amarelo >= 5100 and auxY == 0):
                    auxY = 1
                    inicio[i] = time.time()
                para = time.time()
                tempo = str(round(para - inicio[i], 5))
                print("tempo: {}\n".format(tempo))
                cv2.rectangle(frame_copy, (x-3,y-3), (x+w+2, y+h+2), sup1[i], 1, cv2.LINE_AA)
                cv2.putText(frame_copy, nome[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sup1[i], 1)
                cv2.putText(frame_copy, tempo, (x+150, y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, sup1[i], 3)
        
    
    corte = cv2.bitwise_and(frame, frame, mask = mascara)
    cv2.imshow("Sinal - Video", frame_copy)
    
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break
    time.sleep(0.0)
    aux3 = aux3 + 1
    
    if (aux3 == 1000):
        break
    
sinal.release()
cv2.destroyAllWindows()

        
        
        
        
        
        