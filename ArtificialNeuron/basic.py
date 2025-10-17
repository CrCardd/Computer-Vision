from typing import List
import random as rd
import math as m

T : float = 0.198201
e : float = m.e
n = 0.2

def sigmoid(s) -> float:
    return 1 / (1 + e ** -s)

def d_sigmoid(y) -> float:
    return y * (1 - y)

def S(x : List[float], w : List[float], b : float) -> int:
    s : int = 0
    for i, j in zip(w,x):
        s += (i*j)
    return s + b

def E(y):
    return 0.5 * (T-y)**2


x = [1.2, 3.1, 0.3, 0.9, 1.0, 3.2, 1.2]
w =  [rd.random() for _ in range(len(x))]
b = rd.random()

for i in range(10000):
    s = S(x,w,b)
    y = sigmoid(s)
    
    print(f"Época {i} | Saída: {y} | Erro: {E(y)}")

    for j in range(len(w)):
        w[j] = w[j] - n * (d_sigmoid(y) * x[j] * (y-T))
    
    b = b - n * (d_sigmoid(y) * 1 * (y-T))

print(w)