#--------------------------------------------------------------------------------------------------
#Computational Neuroscience Assignment: Regression, Classification, and Neural Networks
#Question 3: Feedforward network trained using gradient descent and backpropagation.
#BIOL 487 / SYDE 552
#Andrew Reeves - 20459205
#--------------------------------------------------------------------------------------------------

import numpy as np
import pickle

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def shuffle_in_unison(a, b):
    """Takes in arrays a and b and shuffles them such that they are both in
    the same random order."""
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

    return a, b

def loadNetwork(filename="Plant_Weights.pkl"):
    """Returns a Network object from a pkl file."""

    file = open(filename,'rb')
    network = pickle.load(file)

    return network

#--------------------------------------------------------------------------------------------------
#Activation functions:
def hid_act(a): return np.tanh(a)
def out_act(a): return a #(1.0/(1.0 + np.exp(-a)))

#Derivatives of the activation functions:
def d_hid_act(z): return (1.0 - z**2)
def d_out_act(z): return np.ones(z.shape) #z*(1.0 - z)

#--------------------------------------------------------------------------------------------------
class Network:
    def __init__(self, layers=[2,10,10,1]):
        #Initialize the neural network. Note: "N" stands for "Number of _".
        self.layers = layers

        self.layers[0]+=1
        self.weights = []
        self.activations = [np.ones(node) for node in self.layers] #TODO why ones?!?

        for i, _ in enumerate(self.layers[:-1]):
            weightMatrix = np.random.rand(self.layers[i], self.layers[i+1])
            self.weights.append(weightMatrix)

        #Keep the deltas that were back-propagated to the input of the network
        self.input_deltas = np.zeros(self.layers[0])
        self.output_deltas = np.zeros(self.layers[-1]) #TODO check that this shouldn't actually be np.zeros(self.layers[0])

    def Classify(self, inputs):
        #Given an input, what does the network output?

        #Load inputs into the input activations, being careful to keep the bias input node's activation at zero.
        self.activations[0][:-1] = inputs

        #Calculate the hidden node activations and then the output activations using numpy matrix multiplication:
        for i, _ in enumerate(self.weights):
            self.activations[i+1] = hid_act(np.dot(np.transpose(self.weights[i]), self.activations[i]))

        return self.activations[-1] #Output of network

    def Backprop(self, output_deltas, learningrate):
        #Use backpropagation to update the weight values.

        # Calculate the deltas
        deltas = [output_deltas]
        for i in range(len(self.activations)-2):                                # Only want to iterate over number of hidden layers
            hidden_delta = d_hid_act(self.activations[-2-i]) * np.dot(self.weights[-i-1], deltas[i])    # Need to go backwards over the activations, skipping the out activation layer
            deltas.append(hidden_delta)

        #Store output and input deltas for use if desired:
        self.output_deltas = output_deltas
        self.input_deltas = np.dot(self.weights[0], deltas[-1])

        # Update weights
        for i in range(len(deltas)):
            shift = np.outer(self.activations[-2-i], deltas[i])
            self.weights[-1-i] += learningrate * shift

    def getOutputDeltas(self, expected_output_labels):
        """Calculates the output deltas to be used in the backpropagation function."""
        output_deltas = (expected_output_labels - self.activations[-1]) * d_out_act(self.activations[-1])
        return output_deltas

    def Learn(self, input_vectors, labels, iterations, learningrate=0.01):
        #Iterate through all of the training data "iterations" number of times. After each point
        #is classified, the output is used for backpropagation to update the weights of the network.

        for i in np.arange(iterations):
            # i = iteration
            #print i 
            for pos in np.arange(len(labels)): #pos is the index of the training data point being classified.
                self.Classify(input_vectors[pos])
                output_deltas = self.getOutputDeltas(labels[pos])
                self.Backprop(output_deltas, learningrate)

    def batchLearn(self, input_vectors, labels, iterations, learningrate=0.01, batch_size=5):
        #Iterate through all of the training data "iterations" number of times in chunks.
        #Each chunk (batch) is of length batch_size. After each batch is run, the average
        #error for those classifications is then backpropagated through the network.

        for i in np.arange(iterations):
            # i = iteration
            #print i

            #randomize the ordering of the training data
            input_vectors, labels = shuffle_in_unison(input_vectors, labels)

            #For a given "iteration", loop through the training set one batch at a time:
            for j in np.arange(0, len(labels), batch_size):
                #one batch worth of training data

                tmp_output_deltas = 0. #This will be an array
                #print "shape", tmp_output_deltas.shape

                for pos in np.arange(j, j+batch_size, batch_size): #pos is the index of the training data point being classified.
                    self.Classify(input_vectors[pos])
                    tmp_output_deltas += self.getOutputDeltas(labels[pos])

                batch_output_deltas = tmp_output_deltas / batch_size #average

                #Backprop the error after the batch has been run:
                self.Backprop(batch_output_deltas, learningrate)

    def saveNetwork(self, filename="Plant_Weights.pkl"):
        """Saves the network's weights to a pkl file."""

        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
