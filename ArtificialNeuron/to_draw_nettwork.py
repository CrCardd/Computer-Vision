import tkinter as tk
from typing import List, Tuple
from random import random, randint
import math
import time


PADDING = 100
CIRCLE_RADIUS = 20
NEURONS_MARGIN_Y = CIRCLE_RADIUS
NEURONS_MARGIN_X = CIRCLE_RADIUS*5
MAX_NEURONS = 10
Y_INIT = PADDING
X_INIT = PADDING

DELAY = 10

NEURON_DISABLE_COLOR = "#424242"
NEURON_ENABLE_COLOR = "#991774"
LINE_COLOR = "#991774"
ERROR_COLOR = "#BD3636"

root = tk.Tk()
root.title("Neural Network")
root.attributes('-fullscreen', True)

SCREEN_W = root.winfo_screenwidth()
SCREEN_H = root.winfo_screenheight()

canvas = tk.Canvas(root, width=SCREEN_W, height=SCREEN_H, bg="black")
canvas.pack()


class Neuron:
    def __init__(self, w : List[float], b : float, lr : float, x : int, y : int, entity_id : int):
        self.w = w
        self.b = b
        self.lr = lr
        self.entity_id = entity_id

        self.X = x
        self.Y = y

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

        nLs = [3,2] # Neuronios por camada
        nLs.append(self.outputs)


        nO = self.features
        for i in range(len(nLs)):
            
            neurons = []
            for j in range(nLs[i]):
                w = [random() for _ in range(nO)]
                b = random()

                x = X_INIT+(i*NEURONS_MARGIN_X)
                y = (SCREEN_H/2 - (nLs[i]*(CIRCLE_RADIUS*2+NEURONS_MARGIN_Y))/2) + (j*(CIRCLE_RADIUS*2+NEURONS_MARGIN_Y))
                id = canvas.create_oval(
                    x - CIRCLE_RADIUS, 
                    y - CIRCLE_RADIUS, 
                    x + CIRCLE_RADIUS, 
                    y + CIRCLE_RADIUS, 
                    fill=NEURON_DISABLE_COLOR, 
                    outline=NEURON_DISABLE_COLOR
                )

                neuron = Neuron(w, b, self.lr, x, y, id)
                neurons.append(neuron)

            nO = nLs[i]
            self.layers.append(neurons)
         
    def predict(self, x : List[float]):

        for layer, i in zip(self.layers, range(len(self.layers))):
            y_results = []        
            for neuron in layer:
                y = neuron.predict(x)

                self.set_color_neuron(neuron, NEURON_ENABLE_COLOR)
                if(i < len(self.layers)-1):
                    self.connect_next_layer(neuron, self.layers[i+1], LINE_COLOR)                    
                self.set_color_neuron(neuron, NEURON_DISABLE_COLOR)

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
                
                self.set_color_neuron(neuron, NEURON_ENABLE_COLOR)
                self.connect_next_layer(neuron, next_layer, LINE_COLOR, ERROR_COLOR)
                self.set_color_neuron(neuron, NEURON_DISABLE_COLOR)
                
                delta = p * (1 - p) * w_error
                self.layers[i][j].fit(x_for[i], x_for[i+1], delta)
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

    def set_color_neuron(self, neuron : Neuron, color : str):
        canvas.itemconfig(
            neuron.entity_id, 
            fill=color,
            outline=color
        )
        root.update()
        root.after(DELAY)

    def connect_next_neuron(self, neuron : Neuron, next_neuron : Neuron, color : str = "#ffffff"):
        line_id = canvas.create_line(
                neuron.X + CIRCLE_RADIUS, 
                neuron.Y, 
                next_neuron.X - CIRCLE_RADIUS, 
                next_neuron.Y, 
                fill=color, 
                width=2
            )
        return line_id
    
    def connect_next_layer(self, neuron : Neuron, next_layer : List[Neuron], color : str = "#ffffff", next_neuron_color = NEURON_DISABLE_COLOR):

        line_ids = []
        for next_neuron in next_layer:
            self.set_color_neuron(next_neuron, next_neuron_color)
            line_id = self.connect_next_neuron(neuron, next_neuron, color)
            self.set_color_neuron(next_neuron, NEURON_DISABLE_COLOR)
            line_ids.append(line_id)

        root.update()    
        root.after(int(DELAY/2))
        for line_id in line_ids:
            canvas.delete(line_id)

        root.update()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, Y = load_iris(return_X_y = True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)



Net = Network(outputs=3, features=4)


root.after(1000, lambda: Net.fit(X_train, Y_train, 1))

root.mainloop()