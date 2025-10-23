from typing import List, Tuple
from random import random, randint
import math

class Neuron:
    def __init__(
            self, w : List[float], 
            b : float, 
            lr : float
        ):
        self.w = w
        self.b = b
        self.lr = lr

    def predict(self, x):
        s : float = 0
        for i in range(len(self.w)):
            s += x[i] * self.w[i]
        s += self.b
        y = self.active(s)
        return y

    def active(self, s : float):
        return 1/(1 + math.e**-s)

    def fit(self, x, t, delta = None):
        if(delta is None):
            y = self.predict(x)
            delta = (y-t) * y*(1-y)

        self.b = self.b - self.lr * (delta)
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - self.lr * (delta * x[i])
        
        return delta

class Network:
    def __init__(self, outputs, features):
        self.layers : List[List[Neuron]] = []
        self.lr = 0.1
        self.outputs = outputs
        self.features = features # quantidade de features do dataset

        nLs = [728] # Neuronios por camada

        nLs.append(self.outputs)

        nO = self.features
        for i in range(len(nLs)):
            
            neurons = []
            
            for j in range(nLs[i]):
                w = [(random() - 0.5) * 2 for _ in range(nO)]
                b = (random() - 0.5) * 2
            
                neuron = Neuron(w, b, self.lr)
                neurons.append(neuron)

            nO = nLs[i]
            self.layers.append(neurons)
         
    def predict(self, x : List[float]):

        for layer, i in zip(self.layers, range(len(self.layers))):
            y_results = []        
            for neuron in layer:
                y = neuron.predict(x)
                y_results.append(y)

            x = y_results

        return y_results

    def transformY(self, y : int):
        a = [0 for _ in range(self.outputs)]
        if(y >= len(a)):
            return a
        a[y] = 1
        return a

    def epoch(self, x : List[float], y : float):
        # Forwardpass
        x_for = [x[:]]
        x_values = x[:]
        for layer, i in zip(self.layers, range(len(self.layers))):
            results = []
            
            for neuron in layer:                
                p = neuron.predict(x_values)
                results.append(p)

            x_values = results[:]
            x_for.append(x_values)

        y : List[float] = self.transformY(y)
        output_error = []

        # Output layer
        output_layer = self.layers[-1]
        for i in range(len(output_layer)):
            p = x_for[-1][i]
            delta = output_layer[i].fit(x_for[-2], y[i])
            output_error.append(delta)

        # Hidden layers
        for i in range(len(self.layers)-2, -1, -1):
            next_layer = self.layers[i+1]
            crr_layer_error = []
            for j in range(len(self.layers[i])):
                p = x_for[i+1][j]
                neuron = self.layers[i][j]

                w_error  = sum(next_layer[k].w[j] * output_error[k] for k in range(len(output_error)))
                
                delta = p * (1 - p) * w_error
                self.layers[i][j].fit(x_for[i], None, delta)
                crr_layer_error.append(delta)

            output_error = crr_layer_error[:]
       
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
            print('score:\t\t', self.scoreAll(x,y))
            print()

    def accuracy(self, x : List[List[float]], y : List[int]):
        accuracy = 0
        for xi, yi in zip(x,y):
            pred = self.predict(xi)
            max_pred_idx = pred.index(max(pred))
            accuracy += 1 if max_pred_idx == yi else 0 
        return accuracy / len(y)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X, Y = load_iris(return_X_y = True)
scaler = MinMaxScaler()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


Net = Network(outputs=3, features=4)

Net.fit(X_train, Y_train, 1)