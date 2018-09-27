import math
import numpy.matlib 
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

class Net2(object):
    countInputNeurons  = None
    countHiddenNeurons = None
    countOutputNeurons = None

    wInputHidden  = None
    wHiddenOutput = None
    oHidden       = None

    activateBias      = True
    activatePrint     = False
    activateCsvImport = False
    activateCsvExport = False

    fileWIH = 'weightsFromInputToHidden.csv'
    fileWHO = 'weightsFromHiddenToOutput.csv'

    testData = np.c_[np.zeros(0), np.zeros(0), np.zeros(0)]
    
    '''
    Create the net with a number of 
    '''
    def __init__(self, cInput = 2, cHidden = 4, cOutput = 1, eta = 0.01):
        # add 1 because Bias-Neuron
        if(self.activateBias):
            cInput += 1
        
        self.countInputNeurons  = cInput
        self.countHiddenNeurons = cHidden
        self.countOutputNeurons = cOutput
        self.eta                = eta

        self.initWeights()
        self.print('\n\n')
        pass
    
    '''
    Get a result from the network. In circle => 0.8; out of circle => 0    
    '''
    def activate(self, input):
        if(self.activateBias):
            input = np.append(input, 1.0)
        input = np.matrix(input).T
        self.printHeader('Layer I: Input')
        self.print(input)

        self.printHeader('Layer J: Input')
        iHidden = np.dot(self.wInputHidden, input)
        self.print(iHidden)

        self.printHeader('Layer J: Output')
        oHidden = self.sigmoidOn(iHidden)
        self.print(oHidden)

        self.printHeader('Layer K: Input')
        iOutput = np.dot(self.wHiddenOutput, oHidden)
        self.print(iOutput)
        
        self.printHeader('Layer K: Output')
        oOutput = self.sigmoidOn(iOutput)
        self.print(oOutput)
        return oOutput
    
    '''
    Initialize the weights between the layers
    '''
    def initWeights(self):
        if(self.activateCsvImport):
            # import the weights from csv files
            self.importByCSV()
        else:
            # create random weights between 0 and 1
            print(self.countInputNeurons)
            print(self.countHiddenNeurons)
            print(self.countOutputNeurons)
            self.wInputHidden  = np.matlib.rand(self.countHiddenNeurons, self.countInputNeurons).round(2) - 0.5
            self.wHiddenOutput = np.matlib.rand(self.countOutputNeurons, self.countHiddenNeurons).round(2) - 0.5
        ''''''
        # print the weights in the console
        self.printHeader('Gewichte I nach J')
        self.print(self.wInputHidden)
        self.printHeader('Gewichte J nach K')
        self.print(self.wHiddenOutput)
        pass
    
    '''
    Import the weights from a csv file.
    '''
    def importByCSV(self):
        self.wInputHidden  = genfromtxt(self.fileWIH, delimiter=',') - 0.5
        self.wHiddenOutput = genfromtxt(self.fileWHO, delimiter=',') - 0.5
        if(self.activateBias == False):
            print(self.wInputHidden)
            self.wInputHidden  = np.delete(np.array(self.wInputHidden), np.s_[2:3], axis=1)
            print(self.wInputHidden)
        pass
    
    '''
    Export the weights to a csv file.
    '''
    def exportToCSV(self):
        numpy.savetxt(self.fileWIH, self.wInputHidden, delimiter=",")
        numpy.savetxt(self.fileWHO, self.wHiddenOutput, delimiter=",")
        pass
    
    '''
    Runs the sigmoid function on a matrix.
    '''
    def sigmoidOn(self, matrix):
        fnSigmoid = np.vectorize(self.sigmoid, otypes=[np.float])
        return fnSigmoid(matrix)
    
    '''
    Sigmoid-function for one element.
    '''
    def sigmoid(self, x):
        return (1 / (1 + np.exp(2* (-x))))
    
    
    def dsigmoidOn(self, matrix):
        fnDsigmoid = np.vectorize(self.dsigmoid, otypes=[np.float])
        return fnDsigmoid(matrix)
    
    def dsigmoid(self, y):
        return y * (1.0 - y)
    
    # ---------------------------------------
    # TEST AREA!
    # ---------------------------------------

    '''
    Generates Test Data
    '''
    def createTestData(self, s = 10000, l = -0.8, h = 0.8, d = 2):
        # generate random data in range l & h with size s
        x = np.random.uniform(low=l, high=h, size=s)
        y = np.random.uniform(low=l, high=h, size=s)
        # round data on decimals d
        x = np.around(x, decimals=d)
        y = np.around(y, decimals=d)
        # create array with results
        z = np.zeros(s)
        # check whether point is in circle
        for n in range(0, s):
            result = x[n] ** 2 + y[n] ** 2
            if(result <= 1):
                z[n] = 0.8
        return np.c_[x, y, z]
    
    def createTestData2(self, s = 10000, l = -0.8, h = 0.8, d = 2):
        # generate random data in range l & h with size s
        x = np.random.uniform(low=l, high=h, size=s)
        y = np.random.uniform(low=l, high=h, size=s)
        # round data on decimals d
        x = np.around(x, decimals=d)
        y = np.around(y, decimals=d)
        return np.c_[x, y]
    
    def learn(self):
        testDataInCircle = self.createTestData(1000)
        testDataMixed    = self.createTestData(100, -2.0, 2.0)
        self.testData = np.concatenate((testDataInCircle, testDataMixed), axis=0)

        for input in self.testData:
            self.learnSpecific(input)
        
        if(self.activateCsvExport):
            # export the weights to csv files
            self.exportToCSV()
        
        pass
    
    def learn2(self, number):
        testDataInCircle = self.createTestData(number)
        testDataMixed    = self.createTestData(number, -2.0, 2.0)
        data = np.concatenate((testDataInCircle, testDataMixed), axis=0)

        self.testData = np.concatenate((self.testData, data), axis=0)

        for input in self.testData:
            self.learnSpecific(input)
        
        if(self.activateCsvExport):
            # export the weights to csv files
            self.exportToCSV()
        pass

    def learnSpecific(self, input):
        # save target
        target = input[2]
        # remove target
        input = input[:-1]
        if(self.activateBias):
            # add bias
            input = np.append(input, 1.0)
        input   = np.matrix(input).T
        self.print('input:\t' + str(input))
        self.print('wIH:\t' + str(self.wInputHidden))
        iHidden = np.dot(self.wInputHidden, input)
        self.print('iHidden:\t' + str(iHidden))
        oHidden = self.sigmoidOn(iHidden)
        self.print('oHidden:\t' + str(oHidden))
        iOutput = np.dot(self.wHiddenOutput, oHidden)
        self.print('iOutput:\t' + str(iOutput))
        oOutput = self.sigmoidOn(iOutput)
        self.print('oOutput:\t' + str(oOutput))
        

        self.print('wInputHidden: ' + str(self.wInputHidden))
        self.print('wHiddenOutput: ' + str(self.wHiddenOutput))
        # BACKPROPAGATION
        # -----------------------------------------------------------------
        # calculate the error
        eOutput = target - oOutput
        self.print('eOutput:\t' + str(eOutput) + ' = ' + str(target) + ' - ' + str(oOutput))
        eHidden = np.dot(np.matrix(self.wHiddenOutput).T, eOutput)
        self.print('eHidden:\t' + str(eHidden) + ' = ' + str(np.matrix(self.wHiddenOutput).T) + ' * ' + str(eOutput))
        ''''''
        # eta * E * Ok * (1 - Ok) *  Oj.  T
        deltaWHO = (self.eta * np.dot(np.dot(eOutput, self.dsigmoid(oOutput)), np.matrix(oHidden).T))
        #print('Delta' + str(deltaWHO))
        #print((np.array(eHidden) * np.array(self.dsigmoidOn(oHidden))))
        # weight from input to hidden
        deltaWIH = (self.eta * np.dot((np.array(eHidden) * np.array(self.dsigmoidOn(oHidden))), np.matrix(input).T))        
        #print('Delta' + str(deltaWIH))
        self.wHiddenOutput = np.array(self.wHiddenOutput) + deltaWHO
        self.wInputHidden  += np.array(deltaWIH)
        self.print('-----------------------------------------------------')
        pass
    
    def print(self, msg):
        if(self.activatePrint):
            print(msg)
        pass
    
    def printHeader(self, headline):
        self.print('--------------------------------------------------\n' + headline + ':\n--------------------------------------------------')
        pass