'''
Created on 30.03.2018

@author: chibi
'''
from array import array
import matplotlib.pyplot as plt
import numpy as np

class EulerMethod(object):
    config:dict = {
        'yTicks': 1,
        'yMin'  : -300,
        'yMax'  : 300,
        'yLabel': 'y',

        'xTicks': 0.1,
        'xMin'  : -10,
        'xMax'  : 10,
        'xLabel': 'x'
    }

    '''
    a function the will be displayed and takes only float param!
    '''
    func = None
    valueIndex = 1
    y:np.array = []
    x:np.array = []
    eulerResult:np.array = None

    def __init__(self, function):
        # set function
        self.func = function
        # set configuration
        plt.yticks = self.config.get('yTicks')
        plt.xlim(self.config.get('yMin'), self.config.get('yMax'))
        plt.xticks = self.config.get('xTicks')
        plt.xlim(self.config.get('xMin'), self.config.get('xMax'))
        plt.ylabel(self.config.get('yLabel'))
        plt.xlabel(self.config.get('xLabel'))

    '''
    does an euler function with the given data
    '''
    def euler(self, x, deltaT, precision):
        # calculate function result
        y = self.func(x)
        self.x.append(x)
        self.y.append(y)
        print('Wert: ', x, ' = ', y)
        # create new x value
        xNew = x + deltaT * y
        self.valueIndex += 1
        if(np.abs(x - xNew) > precision):
            self.euler(xNew, deltaT, precision)
        else:
            self.eulerResult = np.array([self.x, self.y])
            print('Wert: ', xNew, ' = ', y)
    
    def getNextStep(self, curStep = 0):
        return curStep + self.config.get('xTicks')
    
    '''
    Build the original function in the given range (no euler function)
    '''
    def getRealFunctionByRange(self, xFrom:float, xTo:float):
        y:np.array = []
        x:np.array = []

        # calculates the values for the function in the range
        for i in np.arange(xFrom, xTo, self.config.get('xTicks')):
            stepResult = self.func(i)
            # checking ranges to optimize view (bc. 0.1 steps makes much data)
            inYAxisRange = self.config.get('yMin') <= stepResult <= self.config.get('yMax')
            inXAxisRange = self.config.get('xMin') <= i          <= self.config.get('xMax')

            # add the data to the axises
            if inYAxisRange and inXAxisRange:
                x.append(i)
                y.append(stepResult)
            
        return np.array([x, y])

    '''
    Add the function to the plot.
    '''
    def addFunctionData(self, data:np.array):
        plt.plot(data[0], data[1])
    
    '''
    Shows the plot with the created data.
    '''
    def showPlot(self):
        plt.show()
