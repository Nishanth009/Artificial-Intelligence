import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from neural_network import FCN

## create dataset
dataset = make_moons(200, True, noise=0.2)

X = dataset[0]
y = dataset[1]

## initialize model

model = FCN(inputs=2, outputs=1, hidden_layers=1, hidden_nodes=[8], 
                     hidden_layer_actv='relu', output_layer_act='sigmoid')


## show model details

model.summary()


## train network

model.fit(X, y.reshape(-1,1),learning_rate=0.001, regularizer=0.01, epochs=8000)

## visualize loss
plt.figure(1)
model.plot_history()

## visualize classification


def plot_decision_regions(X, y,W2,W1,b2,b1,res=0.02):
   
    
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res),np.arange(x2_min, x2_max, res))
    newx = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = model.predict(newx)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.ylim(xx2.min(), xx2.max())
    plt.scatter(X[y == 0, 0],X[y == 0, 1],alpha=0.8, c='red',
                              marker='s', label=0)
    plt.scatter(X[y == 1, 0],X[y == 1, 1],alpha=0.8, c='blue',
                              marker='x', label=1)
    plt.show()



plt.figure(2)
plot_decision_regions(X, y.flatten(),model.weights[-1],model.weights[0],model.bias[-1],model.bias[0],res=0.02)
plt.show()






