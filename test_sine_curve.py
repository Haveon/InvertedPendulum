import numpy as np
import matplotlib.pyplot as mpl
from Network import Network

#-------------------------------------------------------------------------------
#Learns to fit to a sine curve.

network = Network(1, 15, 1) #N_inputs=3, N_hidden=10, N_outputs=2
input_vector = 2. * np.pi * (np.random.random([1000,1])-0.5)
labels = 4.*np.sin(2.*input_vector)+0.5

network.Learn(input_vector, labels, 100)
t = np.linspace(-3.14,3.14)
tmps = [network.Classify([t[_]]) for _ in range(50)]
mpl.figure(0)
mpl.plot(input_vector, labels, 'o')
mpl.plot(t,tmps,'r')
mpl.show()
