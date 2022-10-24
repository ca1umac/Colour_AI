import numpy as np


# useful functions
def sigmoid(x):
    z = np.exp(-x)
    sig = 1/(1+z)
    return sig


class Network:
    def __init__(self):
        self.weights = 0
        self.biases = 0

    def feed_forward(self, oldActuation):
        for w, b in self.weights, self.biases:
            newActuation = sigmoid(np.dot(w, oldActuation) + b)
            return newActuation

    def back_propagation(self):
        pass

    def cost_function(self):
        pass


myNetwork = Network()
