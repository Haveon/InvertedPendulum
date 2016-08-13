#--------------------------------------------------------------------------------------------------
#Computational Neuroscience Assignment: Regression, Classification, and Neural Networks
#Question 3: Feedforward network trained using gradient descent and backpropagation.
#BIOL 487 / SYDE 552
#Andrew Reeves - 20459205
#--------------------------------------------------------------------------------------------------

import numpy as np

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


	def Backprop(self, targets, learningrate):
		#Use backpropagation to update the weight values.

		#Calculate the deltas:
		output_deltas = (targets - self.acts_out) * d_out_act(self.acts_out)
		hidden_deltas_2 = d_hid_act(self.acts_hid_2) * np.dot(self.weightsOut, output_deltas)
		hidden_deltas_1 = d_hid_act(self.acts_hid_1) * np.dot(self.weightsMiddle, hidden_deltas_2)

		#Update the output weights:
		shift = np.outer(self.acts_hid_2, output_deltas)
		self.weightsOut += learningrate * shift

		#Update the middle weights:
		shift = np.outer(self.acts_hid_1, hidden_deltas_2)
		self.weightsMiddle += learningrate * shift

		#Update the input weights:
		shift = np.outer(self.acts_in, hidden_deltas_1)
		self.weightsIn += learningrate * shift


	def Learn(self, input_vectors, labels, iterations, learningrate=0.01):
		#Iterate through all of the training data "iterations" number of times. After each point
		#is classified, the output is used for backpropagation to update the weights of the network.

		#TODO: Add batches (batch-training)

		for i in np.arange(iterations):
			print i
			for pos in np.arange(len(labels)): #pos is the index of the training data point being classified.
				self.Classify(input_vectors[pos])
				self.Backprop(labels[pos], learningrate)
