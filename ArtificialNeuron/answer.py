from random import random, randint
from math import exp, sqrt
from matplotlib import pyplot as plt

class NeuralNet:
  def __init__(self):
    self.layers = []
	
	# Primeira Camada
    w = [[random() for i in range(4)] for j in range(2)]
    b = [random() for i in range(2)]
    self.layers.append((w, b))
    
	# Segunda Camada
    w = [[random() for i in range(2)] for j in range(3)]
    # TODO  
    self.layers.append((w, b))
  
  def sigmoid(self, x): 
    # TODO
    pass
  
  def predict(self, X):
    y = X
    for layer in self.layers:
      w, b = layer
      output = []
      for wj, bj in zip(w, b):
    	# TODO
    	# TODO 
        pass
      
      y = output # passando pra próxima layer
    return y
  
  def transformY(self, y):
    if y == 0:
      return [1, 0, 0]
    # TODO
    # TODO
    # TODO
    # TODO
  
  def score(self, X, Y):
    Y = self.transformY(Y)
    Yn = self.predict(X)
    error = sum([(yi - y) ** 2 for yi, y in zip(Y, Yn)])
    return sqrt(error)
  
  def scoreAll(self, X, Y):
    loss = 0
    for x, y in zip(X, Y):
      loss += self.score(x, y)
    return loss / len(Y)
  
  def fit(self, X, Y, epochs = 100):
    scores = []
    for i in range(epochs):
      print("iniciando epoca", i + 1)
      # TODO

      score = self.scoreAll(X, Y)
      scores.append(score)
      print("score", score)
      print()
    return scores

  def epoch(self, X, Y):
    # TODO
    x = X[batch]
    y = Y[batch]
	
    y = self.transformY(y)
    pred = self.predict(x)

    helper = []
    w, b = self.layers[1]
    for j in range(len(b)):
      yj = pred[j]
      # TODO 
      b[j] -= 0.25 * dB
      for i in range(len(w[j])):
        helper.append(dB * w[j][i] * x[i] * w[j][i])
      	# TODO  
    self.layers[1] = (w, b)

    w, b = self.layers[0]
    for j in range(len(b)):
      dB = 0
      yj = self.sigmoid(sum([wj * xj for wj, xj in zip(w[j], x)]) + b[j])
      for lastDevs in helper:
        dB += helper[j] * yj
      # TODO
      for i in range(len(w[j])):
        w[j][i] -= 0.25 * dB * w[j][i] * x[i]
    self.layers[0] = (w, b)
  
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, Y = load_iris(return_X_y = True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# TODO
mode = 
scores = model.fit(X_train, Y_train)
model.scoreAll(X_test, Y_test)

print(scores)
plt.plot(scores)