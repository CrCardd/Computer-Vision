import cv2
import time
import numpy as np
from threading import Thread

pac = cv2.VideoCapture("pacman.mp4")

kernel = np.ones((2,2),np.uint8)

aux = 0

while (True):
    ret, frame = pac.read()
    frame1 = frame[0:342,0:480]
    frame_copy = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    
    #theta = np.acos()
    
    
    
    
    #BGR
    #inf = (0, 150, 210)
    #sup = (50, 255, 255)

    #HSV
    inf = (30, 130 , 150)
    sup = (40, 255, 255)
    
    mascara = cv2.inRange(frame, inf, sup)
    fechado = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    aberto = cv2.morphologyEx(fechado, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    c, h = cv2.findContours(fechado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnts in c:
        (x, y, w, h) = cv2.boundingRect(cnts)
        cv2.rectangle(frame_copy, (x-3,y-3), (x+w+2, y+h+2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_copy, "PACMAN", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
    corte = cv2.bitwise_and(frame, frame, mask = mascara)
    
    cv2.imshow("Pacman - Video", frame_copy)
    
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break
    
    time.sleep(0.03)
    
    aux = aux + 1
    if (aux == 311):
        break
pac.release()
cv2.destroyAllWindows()

        
        
        
        
        
        