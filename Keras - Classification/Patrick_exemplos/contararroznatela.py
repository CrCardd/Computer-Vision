import cv2
import time
import numpy as np
from threading import Thread


kernel = np.ones((2,2),np.uint8)
aux = 0
rice1 = cv2.imread("arroz.jpg")
rice = cv2.resize(rice1, None, fx = 0.6, fy = 0.6)
cv2.imshow("arroz", rice)

rice_1 = rice.copy()

inf = (180, 200, 200)
sup = (255, 255, 255)

mascara = cv2.inRange(rice, inf, sup)
aberto = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations = 2)

c, h = cv2.findContours(aberto, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnts in c:
        aux = aux + 1
        (x, y, w, h) = cv2.boundingRect(cnts)
        print("X: {} e Y: {} - aux: {}".format(x, y, aux))
        cv2.rectangle(rice_1, (x-3,y-3), (x+w+2, y+h+2), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(rice_1, "arroz {}".format(aux), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1)

cv2.imshow("arroz", rice_1)

cv2.waitKey(0)
cv2.destroyAllWindows()