from typing import List, Tuple
from random import random, randint
import math

class Network:
    def __init__(self):
        self.layers : List[Tuple[List[List[float]],float]] = []
        self.lr = 0.1

        n = 4 # quantidade de features do dataset

        nL1 = 2 # quantidade de neuronios da primeira camada
        w = [[random() for i in range(n)] for _ in range(nL1)]
        b = [random() for _ in range(nL1)]
        self.layers.append((w,b))

        # nL2 = 6
        # w = [[random() for i in range(nL1)] for _ in range(nL2)]
        # b = [random() for _ in range(nL2)]
        # self.layers.append((w,b))
       
        nL3 = 3
        w = [[random() for i in range(nL1)] for _ in range(nL3)]
        b = [random() for _ in range(nL3)]
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

    def epoch(self, x : List[float], y : float):
        # Forwardpass
        x_for = [x[:]]
        x_values = x[:]
        for layer in self.layers:
            results = []
            w, b = layer
            for i in range(len(w)):
                s = sum(w[i][j]*x_values[j] for j in range(len(x_values))) + b[i]
                results.append(self.sigmoid(s))
            x_values = results[:]
            x_for.append(x_values)

        y : List[float] = self.transformY(y)
        output_error = []

        # Output layer
        w, b = self.layers[-1]
        for j in range(len(w)):
            p = x_for[-1][j]

            delta = (p-y[j]) * p*(1-p)
            output_error.append(delta)

            b[j] = b[j] - self.lr * (delta)
            for k in range(len(w[j])):
                w[j][k] = w[j][k] - self.lr * (delta * x_for[-2][k])
        self.layers[-1] = (w,b)

        # Hidden layers
        for i in range(len(self.layers)-2, -1, -1):
            w, b = self.layers[i]
            next_ws, _ = self.layers[i+1]
            crr_layer_error = []
            for j in range(len(b)):
                p = x_for[i+1][j]
                
                w_error  = sum(next_ws[k][j] * output_error[k] for k in range(len(output_error)))
                delta = p * (1 - p) * w_error
                crr_layer_error.append(delta)
                
                b[j] = b[j] - self.lr * delta
                for k in range(len(w[j])):
                    w[j][k] = w[j][k] - self.lr * (delta * x_for[i][k])
            output_error = crr_layer_error[:]
            self.layers[i] = (w,b)
       
    def score(self, x : List[float], y : float):
        r : List[float] = self.transformY(y)
        y : List[float] = self.predict(x)
        score = sum((ri-yi)**2 for ri, yi in zip(r,y))

        return score**0.5
    
    def scoreAll(self, x : List[List[float]], y : List[float]):

        score = 0
        for xi, yi in zip(x,y):
            score += self.score(xi, yi)
        return score / len(y)

    def fit(self, x : List[List[float]], y = List[float], epochs : int = 100):

        
        for i in range(epochs):
            print(f'starting epoch\t [{i+1}]')
            for xj, yj in zip(x,y):
                self.epoch(xj,yj)
            # self.epoch(x[randint(0,len(y)-1)], y[randint(0,len(y))-1])
            print('score:\t\t', self.scoreAll(x,y))
            print()





from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, Y = load_iris(return_X_y = True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

Net = Network()
Net.fit(X_train, Y_train,1)
a = Net.scoreAll(X_test, Y_test)
print(a)