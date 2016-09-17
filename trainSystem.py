import numpy as np
from matplotlib import pyplot as plt
from Network import Network

def systemDE(y,t):
    """
    The non linearized equation for a pendulum with a control term:
    d2y/dt2 = -sin(y)
    """
    return -np.sin(y)

def verletStep(f, theta_last, theta_now, control, t, dt):
    """
    Take a single verlet step for the given function.
    This is coded for this problem specifically... not general.
    """
    return 2*theta_now - theta_last + (f(theta_now,t)+control)*dt**2

def generateInputs(numInputs, dt=0.001, f=systemDE):
    """
    Generate a set of inputs for the system network.
    Returns an array that has shape (numInputs,3).
    The first column is theta_{n-1}, second row is theta_\{n\}.
    The second column is calculated by picking a theta_dot such that the pendulum
    won't be able to travel more than a quater revolution.
    The last column is a control signal, a random value between -1 and 1.
    """
    thetas = np.random.random([numInputs,1])*2*np.pi

    theta_dots = (np.random.random([numInputs,1])-0.5)*np.pi*dt
    forward_thetas = (thetas + theta_dots*dt + 0.5*f(thetas ,None)*dt**2)%(2*np.pi)

    control = 2*(np.random.random([numInputs,1])-0.5)

    return np.concatenate([thetas,forward_thetas,control], axis=1)

def meanUniformDistribution(minimum, maximum):
    return (maximum+minimum)/2.

def stdUniformDistribution(minimum, maximum):
    return (maximum - minimum)/(2*np.sqrt(3))

def normalizeInputs(inputs):
    """
    This function changes inputs in place.

    It finds the mean and standard deviation of each column and makes the mean
    zero and standard deviation one.
    It returns inputs, and a settings list which has the mean and std for each
    column.
    """
    settings = []
    for i in range(inputs.shape[1]):
        data = inputs[:,i]
        mean = np.mean(data)
        std  = np.std(data-mean)
        inputs[:,i] = (data-mean)/std
        settings.append((mean,std))
    return inputs, settings

def generateOutputs(inputs, dt=0.001, f=systemDE):
    """
    Given a set of inputs it computes the outputs we expect to get out of the
    neural net using the verletStep function.
    Returns the un-normalized outputs.
    """
    theta_last = inputs[:,0]
    theta_now  = inputs[:,1]
    control    = inputs[:,2]
    outputs    = verletStep(f, theta_last, theta_now, control, None, dt)%(2*np.pi)
    return outputs

def classificationError(neuralNet, normedInputs, normedOutputs):
    """Calculates the classification error of the NN given a set of inputs and
    expected outputs."""
    number_of_samples = len(normedOutputs)
    tmp_error = 0.

    for i in xrange(number_of_samples):
        classification = neuralNet.Classify(normedInputs[i])
        print classification, normedOutputs[i]
        tmp_error += np.abs((classification - normedOutputs[i])/normedOutputs[i])

    error = tmp_error / number_of_samples
    return error[0]

if __name__ == '__main__':
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

    net = Network(3,10,1)
    net.Learn(normedInputs, normedOutputs, 1000, learningrate=0.01)
