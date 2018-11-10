import numpy as np
import matplotlib.pyplot as plt

class FCN:

    def __init__(self, inputs=2, outputs=1, hidden_layers=1, hidden_nodes=[2], 
                     hidden_layer_actv='relu', output_layer_act='linear'):
        
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.activation_hidden = hidden_layer_actv
        self.activation_output = output_layer_act
        self.params_init()
        
    
    def summary(self):
        print "input layer nodes {0}".format(self.input_nodes)
        print "output layer nodes {0}".format(self.output_nodes)
        print "number of hidden layers {0}".format(self.hidden_layers)
        print "number of nodes in each hidden layer {0}".format(self.hidden_nodes)
        print "hidden_layer_activation {0}".format(self.activation_hidden)
        print "output_layer_activation {0}".format(self.activation_output)
        
    
    def params_init(self):
        self.weights = list()
        self.bias = list()
        if self.hidden_layers != len(self.hidden_nodes):
            print ("invalid hidden nodes")
            return
        for i in range(len(self.hidden_nodes)+1):
            if i==0:
                W = np.random.randn(self.input_nodes, self.hidden_nodes[i]) * np.sqrt(self.input_nodes)
            elif i==len(self.hidden_nodes):
                W = np.random.randn(self.hidden_nodes[i-1], self.output_nodes) * np.sqrt(self.hidden_nodes[i-1])
            else:
                W = np.random.randn(self.hidden_nodes[i-1],self.hidden_nodes[i]) * np.sqrt(self.hidden_nodes[i-1])
            b = np.zeros((1,W.shape[1])) 
            self.weights.append(W)
            self.bias.append(b)
            
    def linear(self, X, W, b):
        z = X.dot(W) + b
        return z
    
    def activation(self, x, layer = 'hidden'):
        
        if layer == 'hidden':
            actv = self.activation_hidden
        else:
            actv = self.activation_output
        
        if actv == 'tanh':
            return np.tanh(x)
        elif actv == 'sigmoid':
            return 1/(1 + np.exp(-x))
        elif actv == 'relu':
            return np.maximum(0, x)
        elif actv == 'linear':
            return x
        
    
    def calc_loss(self, predicted, y):
        m = predicted.shape[0]
        error = (predicted - y)
        return error
    
    def forward(self, X):
        self.layer_outs = list()
        for i, W in enumerate(self.weights):
            X = self.linear(X, W, self.bias[i])
            if i==(len(self.weights)-1):
                X = self.activation(X, layer = 'output')
            else:
                X = self.activation(X)
            self.layer_outs.append(X)
        return X
    
    def back_prop(self, error, lr, reg):
        for i in range(len(self.weights)):
            
            layer_no = (len(self.weights)-1)-i
            layer_op_curr = self.layer_outs[layer_no]
            
            if layer_no == (len(self.weights)-1):
                if self.activation_output == 'sigmoid':
                    hidden_outs = np.multiply(layer_op_curr, (1 - layer_op_curr))
                elif self.activation_output == 'tanh':
                    hidden_outs = 1 - np.power(layer_op_curr, 2) 
                elif self.activation_output == 'linear':
                    hidden_outs = 1
            else:
                if self.activation_hidden == 'sigmoid':
                    hidden_outs = np.multiply(layer_op_curr, (1 - layer_op_curr))
                elif self.activation_hidden == 'tanh':
                    hidden_outs = 1 - np.power(layer_op_curr, 2) 
                elif self.activation_hidden == 'relu':
                    layer_op_curr[layer_op_curr < 0] = 0 
                    layer_op_curr[layer_op_curr > 0] = 1 
                    hidden_outs = layer_op_curr
                
            if (layer_no == 0):
                layer_op_prev = self.data
            else:
                layer_op_prev = self.layer_outs[layer_no - 1]
            
            error_prop = np.multiply(error, hidden_outs)
            dW = layer_op_prev.T.dot(error_prop)
            dW += reg * self.weights[layer_no]
            dB = np.sum(error_prop, axis=0, keepdims=True)    
            error = np.dot(error_prop, self.weights[layer_no].T)
            self.weights[layer_no] -= lr * dW
            self.bias[layer_no] -= lr * dB
    
            
    def fit(self, X, Y, learning_rate=0.1, regularizer=0.01, history=True, epochs=1):
        self.data = X
        self.hist = []
        self.epochs = epochs
        for i in range(epochs):
            yhat = self.forward(X)
            error = self.calc_loss(yhat, Y)
            loss = np.mean(np.power(error,2))
            self.hist.append(loss)
            print "loss:",loss
            self.back_prop(error, learning_rate, regularizer)
    
    def plot_history(self):
        plt.plot(range(self.epochs), self.hist)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()
    
    def predict(self, X):
        return self.forward(X)
            
