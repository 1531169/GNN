import math
import numpy

# BIAS
bias1 = -4
bias2 =  4
# WEIGHTS
w11   =   6
w21   =  10
w12   = -10
w22   =   0
# START VALUES
o1    = 0.770719
o2    = 0.00978897

# WEIGHTS as numpy array/matrix
weights = numpy.array([[w11, w12], [w21, w22]])
# PRECISION for the while-loop
epsilon = 0.0001

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# calculate the eigenvalus for the weights and the scale factor
def getEigenvalues(weights, scalefactor):
    return numpy.array(numpy.real(numpy.linalg.eigvals(weights * scalefactor)))

# checks whether the changed values changed in the right way
def isChangeHigherThan(epsilon):
    # changed the inputs with a precision of epsilon?
    isDiff1 = math.fabs(copyInput1 - input1) > epsilon
    isDiff2 = math.fabs(copyInput2 - input2) > epsilon
    # changed the eigenvalues with a precision of epsilon?
    isDiff3 = math.fabs(eigenvalues[0] - 1) >= epsilon
    isDiff4 = math.fabs(eigenvalues[1] - 1) >= epsilon
    return isDiff1 or isDiff2 or isDiff3 or isDiff4

def getOutput(in1, in2, bias, iW1, iW2, iW3):
    return in1 * weights[iW1][iW1] + in2 * weights[iW2][iW2] + bias

# INPUTS
input1 = 0
input2 = 0
# TEMP-INPUTS
copyInput1 = 0
copyInput2 = 0

# SCALING FACTOR for the weights
scalingFactor      = 1.0
# SCALING CHANGE FACTOR (to change the scaling factor)
changePerIteration = 0.0001
# EIGENVALUES of the weights
eigenvalues        = getEigenvalues(weights, scalingFactor)

while isChangeHigherThan(epsilon):
    input1 = sigmoid(o1 * weights[0][0] + o2 * weights[1][0] + bias1)
    input2 = sigmoid(o2 * weights[1][1] + o1 * weights[0][1] + bias2)

    # run 100 times
    for x in range(0, 100):
        # put in temp values to simulate synchron
        copyInput1 = input1
        copyInput2 = input2
        input1 = sigmoid(copyInput1 * weights[0][0] * scalingFactor + copyInput2 * weights[1][0] * scalingFactor + bias1)
        input2 = sigmoid(copyInput2 * weights[1][1] * scalingFactor + copyInput1 * weights[0][1] * scalingFactor + bias2)
    
    print(copyInput1, copyInput2)
    print(input1, input2)
    print("Skalierungsfaktor:\t", round(scalingFactor, 5))
    print("Beträge:\t\t",         math.fabs(copyInput1 - input1), math.fabs(copyInput2 - input2))
    print("Eingenwert:\t\t",      eigenvalues)
    print("===============================================================")
    # calculate the new eigenvalues      
    eigenvalues = getEigenvalues(weights, scalingFactor)
    # change factor
    scalingFactor -= changePerIteration

print("Shotgun fertigggggg.")
print("Finale Eigenwerte:\n", eigenvalues)
print("Finale Gewichte:\n", weights * scalingFactor)