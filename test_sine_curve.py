import numpy as np
import matplotlib.pyplot as mpl
from Network import Network, loadNetwork

#-------------------------------------------------------------------------------
#Learns to fit to a sine curve.

np.random.seed(1)
network = Network([1, 15, 15, 1]) #N_inputs=3, N_hidden=10, N_outputs=2
input_vector = 2. * np.pi * (np.random.random([1000,1])-0.5)
labels = 4.*np.sin(2.*input_vector)+0.5

#network.Learn(input_vector, labels, 100) #Learn by looking at one training point,label at a time
network.Learn(input_vector, labels, 400, learningrate=0.001)
t = np.linspace(-3.14,3.14)
tmps = [network.Classify([t[_]]) for _ in range(len(t))]

print """ "Input error" (deltas backpropagated through the whole network):  """
print network.input_deltas

mpl.figure(0)
mpl.plot(input_vector, labels, 'o', label='Training data')
mpl.plot(t,tmps,'r', label='Prediction')
mpl.legend()
mpl.show()
