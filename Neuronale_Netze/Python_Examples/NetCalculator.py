import threading
import time

import numpy as np
import matplotlib.pyplot as plt

import math
import numpy.matlib 
from numpy import genfromtxt

class NetCalculator(threading.Thread):

    countInputNeurons  = None
    countHiddenNeurons = None
    countOutputNeurons = None

    wInputHidden  = None
    wHiddenOutput = None
    oHidden       = None

    activateBias        = True
    activatePrint       = False
    activateCsvImport   = False
    activateCsvExport   = False
    activateStartUpData = True

    fileWIH = 'weightsFromInputToHidden.csv'
    fileWHO = 'weightsFromHiddenToOutput.csv'

    displaySeries = np.c_[np.zeros(0), np.zeros(0)]
    outputSeries = np.c_[np.zeros(0)] #np.c_[np.zeros(0), np.zeros(0)]
    displayErrorSeries = np.c_[np.zeros(0), np.zeros(0)]
    outputErrorSeries  = np.c_[np.zeros(0)]
    testData     = np.c_[np.zeros(0), np.zeros(0), np.zeros(0)]
    learnIndex   = 0

    exitFlag = 0

    def __init__(self, cInput = 2, cHidden = 5, cOutput = 1, eta = 0.2):
        threading.Thread.__init__(self)

        # add 1 because Bias-Neuron
        if(self.activateBias):
            cInput += 1
        
        self.countInputNeurons  = cInput
        self.countHiddenNeurons = cHidden
        self.countOutputNeurons = cOutput
        self.eta                = eta

        self.initWeights()
        self.print('\n\n')

        # init test data
        if self.activateStartUpData:
            testDataInCircle = self.createTestData(0)
            # testDataMixed    = self.createTestData(20000, -1.5, 1.5) # hierbei lernt es etwas mit lernrate 0.2
            testDataMixed    = self.createTestData(20000, -2, 2)
            self.testData    = np.concatenate((testDataInCircle, testDataMixed), axis=0)
        pass
    
    def run(self):
        while not self.exitFlag:
            self.update()
            time.sleep(0.1)
            pass
        pass
    
    def update(self):
        print('Calculating...')
        self.bulkLearn(20)
    
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

    forIndex = 0

    def bulkLearn(self, bulk):
        if self.learnIndex > len(self.testData):
            return

        for input in self.testData[self.learnIndex:self.learnIndex+bulk,:]:
            input[2] = self.learnSpecific(input)
            # output = self.learnSpecific(input) # same like line above
            # print(str(self.testData[self.forIndex]))
            # self.testData[self.forIndex,2] = output
            self.forIndex += 1
        
        
        self.learnIndex += bulk

        if(self.activateCsvExport):
            # export the weights to csv files
            self.exportToCSV()
        pass
    
    index = 0

    sumForMiddle    = 0
    countForMiddle  = 0
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
        # Bias for hidden layer
        oHidden[:1] = 1
        #print('oHidden:\t' + str(oHidden) + ' | ' + str(oHidden[:1]))
        self.print('oHidden:\t' + str(oHidden))
        iOutput = np.dot(self.wHiddenOutput, oHidden)
        self.print('iOutput:\t' + str(iOutput))
        oOutput = self.sigmoidOn(iOutput)

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
        self.wHiddenOutput += deltaWHO
        self.wInputHidden  += np.array(deltaWIH)
        self.print('-----------------------------------------------------')

        # never more than 500 items...
        if(len(self.outputSeries) >= 500):
            self.outputSeries = self.outputSeries[+1:]
        if(len(self.outputErrorSeries) >= 500):
            self.outputErrorSeries = self.outputErrorSeries[+1:]
        self.index += 1 # unused
        self.outputSeries = np.concatenate((self.outputSeries, oOutput), axis=0)
        self.displaySeries = np.array(self.outputSeries.T)[0]
        #calc middle error
        self.sumForMiddle += (eOutput ** 2)
        self.countForMiddle += 1
        middleValue = self.sumForMiddle / self.countForMiddle
        self.outputErrorSeries  = np.concatenate((self.outputErrorSeries, middleValue), axis=0)
        self.displayErrorSeries = np.array(self.outputErrorSeries.T)[0]

        self.print('OUTPUT:\t' + str(oOutput))
        self.print('FEHLER:\t' + str(eOutput))
        return oOutput
    
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
            self.wInputHidden  = (np.matlib.rand(self.countHiddenNeurons, self.countInputNeurons).round(2) - 0.5) / 10
            self.wHiddenOutput = (np.matlib.rand(self.countOutputNeurons, self.countHiddenNeurons).round(2) - 0.5) / 10
        ''''''
        # print the weights in the console
        self.printHeader('Gewichte I nach J')
        print(self.wInputHidden)
        self.printHeader('Gewichte J nach K')
        print(self.wHiddenOutput)
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
    
    def print(self, msg):
        if(self.activatePrint):
            print(msg)
        pass
    
    def printHeader(self, headline):
        self.print('--------------------------------------------------\n' + headline + ':\n--------------------------------------------------')
        pass