import numpy as np
import matplotlib.pyplot as mpl
from Network import Network

#-------------------------------------------------------------------------------
#Learns to fit to two superimposed sine curves with noise.

network = Network(1, 20, 1) #N_inputs=3, N_hidden=10, N_outputs=2
input_vector = 2. * np.pi * (np.random.random([1000,1])-0.5)
labels = (4.*np.sin(2.*input_vector) + 0.5 + np.cos(6.*input_vector)) + 0.25*np.random.normal(0.5, 1., [len(input_vector),1])

network.batchLearn(input_vector, labels, 500, learningrate=0.1)
t = np.linspace(-3.14, 3.14, 100)
tmps = [network.Classify([t[_]]) for _ in range(len(t))]

print """ "Input error" (deltas backpropagated through the whole network):  """
print network.input_deltas

#Plot the things
mpl.figure(0)
mpl.title('Neural Network - Prediction')
mpl.plot(input_vector, labels, 'o', label='Training points')
mpl.plot(t,tmps,'r', label='Predicted signal')
mpl.plot(t, (4.*np.sin(2.*t) + 0.5 + np.cos(6.*t)), label='Noise-free signal')
mpl.legend()
mpl.show()
