import numpy as np
import matplotlib.pyplot as mpl
from Network import Network
from trainSystem import *

#-------------------------------------------------------------------------------
#Set up training data

DEnet = Network([3, 15, 15, 1])

#-------------------------------------------------------------------------------
#Training / learning

#DEnet.Learn(normedInputs, normedOutputs, 10, learningrate=0.01)

iterations = 13
training_data_errors = []
evaluation_data_errors = []
#learningrates = [1E-2, 1E-2, 5E-3, 1E-3, 5E-4, 1E-4, 5E-5, 5E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5]
learningrates = [1E-2, 1E-2, 1E-2, 5E-3, 5E-3, 5E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 5E-4, 5E-4]

for i in xrange(iterations):
    inputs = generateInputs(1e5)
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
    #Set up evaluation data for checking the performance of the trained NN

    evaluation_inputs = generateInputs(1e4)
    evaluation_outputs= generateOutputs(evaluation_inputs)

    evaluation_meanTheta = meanUniformDistribution(0, 2*np.pi)
    evaluation_meanControl= meanUniformDistribution(-1, 1)
    evaluation_means = np.array([evaluation_meanTheta,evaluation_meanTheta,evaluation_meanControl])

    evaluation_stdTheta = stdUniformDistribution(0, 2*np.pi)
    evaluation_stdControl = stdUniformDistribution(0, 2*np.pi)
    evaluation_stds = np.array([evaluation_stdTheta,evaluation_stdTheta,evaluation_stdControl])

    evaluation_normedInputs = (evaluation_inputs - evaluation_means)/evaluation_stds
    evaluation_normedOutputs = (evaluation_outputs - evaluation_meanTheta)/evaluation_stdTheta



    learningrate_tmp = learningrates[i]
    DEnet.batchLearn(normedInputs, normedOutputs, 1, learningrate=learningrate_tmp,batch_size=1)
    #DEnet.Learn(normedInputs, normedOutputs, 1, learningrate=learningrate_tmp)
    print "*Iteration ", i
    print "Classification errors:"
    training_data_errors.append(classificationError(DEnet, normedInputs, normedOutputs)*100)
    print "-Training data:", training_data_errors[i], '%'
    evaluation_data_errors.append(classificationError(DEnet, evaluation_normedInputs, evaluation_normedOutputs)*100)
    print "-Evaluation data:", evaluation_data_errors[i], '%'

post_training_error_training_data = classificationError(DEnet, normedInputs, normedOutputs)*100

#-------------------------------------------------------------------------------
#Output results

post_training_error_evaluation_data = classificationError(DEnet, evaluation_normedInputs, evaluation_normedOutputs)*100

print "Pre-training error on training data", pre_training_error_training_data, "%"
print "Post-training error on training data:", post_training_error_training_data, "%"
print "Post-training error on evaluation data:", post_training_error_evaluation_data, "%"

"Arrays of training and evaluation data errors, respectively:"
print training_data_errors
print evaluation_data_errors

DEnet.saveNetwork()
