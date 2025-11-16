import cv2
import time
import numpy as np

sinal = cv2.VideoCapture("project.mp4")

kernel = np.ones((3,3),np.uint8)

aux = 0

#BGR
inf = (140, 140, 140)
sup = (255, 255, 255)
    
limInfX  =     np.array([100, 153, 212,  41, 136, 200])
limSupX  =     np.array([153, 212, 270, 136, 220, 320])

limInfY  =     np.array([104, 104, 104, 164, 164, 164])
limSupY  =     np.array([164, 164, 164, 250, 250, 250])

limInfAreas  = np.array([160, 160, 160, 400, 400, 400])
limSupAreas  = np.array([350, 350, 350, 800, 800, 800])

cores  =       ((0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 0,255), (0, 0,255), (0, 0,255))

temp =         np.array([ 0, 0, 0, 0, 0, 0])

contador = 0
while (True):

    ret, frame1 = sinal.read()
    frame = cv2.resize(frame1, None, fx = 1.6, fy = 1.6)
    frame_copy = frame.copy()
    
    frameGauss = cv2.GaussianBlur(frame, (5, 5), 4)
    mascara = cv2.inRange(frameGauss, inf, sup )
    fechado = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations = 10)
    #dilatar = cv2.dilate(fechado, kernel, iterations = 10)
    c, h = cv2.findContours(fechado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
    for cont in c:
        M = cv2.moments(cont)
        (x, y, w, h) = cv2.boundingRect(cont)
        
        for i in range(6): #1
            if (x > limInfX[i] and x < limSupX[i] and y > limInfY[i] and y < limSupY[i]):
                if (M['m00'] >= limInfAreas[i] and M['m00'] <= limSupAreas[i]):
                    temp[i] = 1
                    if (i > 2):
                        if (temp[i]==1 and temp[i-3]==0):
                            temp[i]=0
                    cv2.rectangle(frame_copy, (x-3,y-3), (x+w+2, y+h+2), cores[i], 1, cv2.LINE_AA)
                    cv2.putText(frame_copy, "carro branco", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cores[i], 1)

                    if (temp[i - 3] == temp[i] and temp[i] > 0):
                        contador += 1
                        temp[i] = 0
                        temp[i - 3] = 0 
            cv2.putText(frame_copy, str(contador), (320, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            
    print("Quantidade de carros {}".format(contador))
    
    corte = cv2.bitwise_and(frame, frame, mask = fechado)
    cv2.imshow("Sinal - Video", frame_copy)
    #cv2.imshow("Sinal - Video 1", corte)
    
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break
    time.sleep(0.07)
    aux = aux + 1

    if (aux == 115):
        break
    
sinal.release()
cv2.destroyAllWindows()

        
        
        
        
        
        