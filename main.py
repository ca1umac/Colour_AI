import numpy as np

# useful functions
def sigmoid(x):
    z = np.exp(-x)
    sig = 1/(1+z)
    return sig


class Network:
    def __init__(self):
        self.weights = np.array()
        self.biases = np.array()
        self.actuations = np.array()
        self.layerinfo = [3,16,16,10]

    def init_arrays(self):
        self.weights = np.loadtxt("weights.csv", delimiter=",")
        self.biases = np.loadtxt("biases.csv", delimiter=",")

    def save_arrays(self):
        np.savetxt("weights.csv", self.weights)
        np.savetxt("biases.csv", self.biases)

    def feed_forward(self, oldActuation):
        for w, b in self.weights, self.biases:
            newActuation = sigmoid(np.dot(w, oldActuation) + b)
        return newActuation

    def back_propagation(self):
        pass

    def particular_cost(self):
        pass

    def cost_function(self):
        pass

    def SGD(self):
        pass


myNetwork = Network()
