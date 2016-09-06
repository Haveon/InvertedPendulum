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
    def __init__(self, N_inputs=2, N_hidden=10, N_outputs=1):
        #Initialize the neural network. Note: "N" stands for "Number of _".
        self.N_in = N_inputs + 1 #Includes a bias weight vector.
        self.N_hid_1 = N_hidden
        self.N_hid_2 = N_hidden
        self.N_out = N_outputs

        #Initialize the weights with random values.
        self.weightsIn = np.random.rand(self.N_in, self.N_hid_1)
        self.weightsMiddle = np.random.rand(self.N_hid_1, self.N_hid_2)
        self.weightsOut = np.random.rand(self.N_hid_2, self.N_out)

        #"acts" = "activations", "in" = "inputs", "hid" = "hidden", "out" = "outputs"
        self.acts_in = np.zeros(self.N_in) + 1
        self.acts_hid_1 = np.zeros(self.N_hid_1) + 1
        self.acts_hid_2 = np.zeros(self.N_hid_2) + 1
        self.acts_out = np.zeros(self.N_out) + 1

        #Keep the deltas that were back-propagated to the input of the network
        self.input_deltas = np.zeros(self.N_in)
        self.output_deltas = np.zeros(self.N_in)

    def Classify(self, inputs):
        #Given an input, what does the network output?

        #Load inputs into the input activations, being careful to keep the bias input node's activation at zero.
        for i in np.arange(len(inputs)):
            self.acts_in[i] = inputs[i]

        #Calculate the hidden node activations and then the output activations using numpy matrix multiplication:
        self.acts_hid_1 = hid_act(np.dot(np.transpose(self.weightsIn), self.acts_in))
        self.acts_hid_2 = hid_act(np.dot(np.transpose(self.weightsMiddle), self.acts_hid_1))
        self.acts_out = out_act(np.dot(np.transpose(self.weightsOut), self.acts_hid_2))

        return self.acts_out #Output of network

    def Backprop(self, output_deltas, learningrate):
        #Use backpropagation to update the weight values.

        #Calculate the deltas:
        hidden_deltas_2 = d_hid_act(self.acts_hid_2) * np.dot(self.weightsOut, output_deltas)
        hidden_deltas_1 = d_hid_act(self.acts_hid_1) * np.dot(self.weightsMiddle, hidden_deltas_2)

        #Store output and input deltas for use if desired:
        self.output_deltas = output_deltas
        self.input_deltas = np.dot(self.weightsIn, hidden_deltas_1)

        #Update the output weights:
        shift = np.outer(self.acts_hid_2, output_deltas)
        self.weightsOut += learningrate * shift

        #Update the middle weights:
        shift = np.outer(self.acts_hid_1, hidden_deltas_2)
        self.weightsMiddle += learningrate * shift

        #Update the input weights:
        shift = np.outer(self.acts_in, hidden_deltas_1)
        self.weightsIn += learningrate * shift

    def getOutputDeltas(self, expected_output_labels):
        """Calculates the output deltas to be used in the backpropagation function."""
        output_deltas = (expected_output_labels - self.acts_out) * d_out_act(self.acts_out)
        return output_deltas

    def Learn(self, input_vectors, labels, iterations, learningrate=0.01):
        #Iterate through all of the training data "iterations" number of times. After each point
        #is classified, the output is used for backpropagation to update the weights of the network.

        for i in np.arange(iterations):
            # i = iteration
            print i
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
            print i

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
