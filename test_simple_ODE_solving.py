import numpy as np
import matplotlib.pyplot as mpl
from Network import Network
from trainSystem import *

#-------------------------------------------------------------------------------
#Set everything up before training

DEnet = Network(3, 10, 1)

inputs = generateInputs(1e3)
outputs= generateOutputs(inputs)

meanTheta = meanUniformDistribution(0, 2*np.pi)
meanControl= meanUniformDistribution(-1, 1)
means = np.array([meanTheta,meanTheta,meanControl])

stdTheta = stdUniformDistribution(0, 2*np.pi)
stdControl = stdUniformDistribution(0, 2*np.pi)
stds = np.array([stdTheta,stdTheta,stdControl])

normedInputs = (inputs - means)/stds
normedOutputs = (outputs - meanTheta)/stdTheta

pre_training_error_training_data = classificationError(DEnet, normedInputs, normedOutputs)*100

#-------------------------------------------------------------------------------
#Training time
DEnet.Learn(normedInputs, normedOutputs, 10, learningrate=0.01)

post_training_error_training_data = classificationError(DEnet, normedInputs, normedOutputs)*100

#-------------------------------------------------------------------------------
#Set up evaluation data and check the performance of the trained NN

inputs = generateInputs(1e3)
outputs= generateOutputs(inputs)

meanTheta = meanUniformDistribution(0, 2*np.pi)
meanControl= meanUniformDistribution(-1, 1)
means = np.array([meanTheta,meanTheta,meanControl])

stdTheta = stdUniformDistribution(0, 2*np.pi)
stdControl = stdUniformDistribution(0, 2*np.pi)
stds = np.array([stdTheta,stdTheta,stdControl])

normedInputs = (inputs - means)/stds
normedOutputs = (outputs - meanTheta)/stdTheta

post_training_error_evaluation_data = classificationError(DEnet, normedInputs, normedOutputs)*100

print "Pre-training error on training data", pre_training_error_training_data, "%"
print "Post-training error on training data:", post_training_error_training_data, "%"
print "Post-training error on evaluation data:", post_training_error_evaluation_data, "%"
