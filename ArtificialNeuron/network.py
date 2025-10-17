from typing import List, Tuple
from random import random
import math

class Network:
    def __init__(self):
        self.layers : List[Tuple[List[List[float]],float]] = []

        n = 4 # quantidade de features do dataset

        nL1 = 3 #quantidade de neuronios da primeira camada
        w = [[random() for i in range(n)] for _ in range(nL1)]
        b = [random() for _ in range(nL1)]
        self.layers.append((w,b))

        nL2 = 3
        w = [[random() for i in range(nL1)] for _ in range(nL2)]
        b = [random() for _ in range(nL2)]
        self.layers.append((w,b))

    def sigmoid(self, s : float):
        return 1/(1 + math.e**-s)
    
    def predict(self, x : List[float]):

        for layer in self.layers:
            w, b = layer
            y_results = []        
            for wj, bj in zip(w,b):
                s : float = 0
                for i in range(len(x)):
                    s += x[i] * wj[i]
                s += bj
                y = self.sigmoid(s)
                y_results.append(y)

            x = y_results

        return y_results

    def transformY(self, y : int):
        a = [0, 0, 0]
        if(y >= len(a)):
            return a
        a[y] = 1
        return a

    def score(self):
        pass
    
    def scoreAll(self):
        pass

    def fit(self):
        pass

    def epoch(self):
        pass
    





from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, Y = load_iris(return_X_y = True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)





print(X_train)
print()
print(Y_train)