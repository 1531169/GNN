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
        'yMin'  : -10,
        'yMax'  : 5,
        'yLabel': 'x (Wert)',

        'xTicks': 0.1,
        'xMin'  : -5,
        'xMax'  : 10,
        'xLabel': 't (Zeit)'
    }

    '''
    a function the will be displayed and takes only float param!
    '''
    func = None

    def __init__(self, function):
        self.func = function
        plt.yticks = self.config.get('yTicks')
        plt.xlim(self.config.get('yMin'), self.config.get('yMax'))
        plt.xticks = self.config.get('xTicks')
        plt.xlim(self.config.get('xMin'), self.config.get('xMax'))
        plt.ylabel(self.config.get('yLabel'))
        plt.xlabel(self.config.get('xLabel'))
    
    def explicitMethod(self, initialConditions:array, deltaT):
        return 0
    
    def getNextStep(self, curStep = 0):
        return curStep + self.config.get('xTicks')
    
    def getRealFunctionByRange(self, xFrom:float, xTo:float):
        y:np.array = []
        x:np.array = []

        for i in np.arange(xFrom, xTo, self.config.get('xTicks')):
            stepResult = self.func(i)
            # checking ranges to optimize view (bc. 0.1 steps makes much data)
            inYAxisRange = self.config.get('yMin') <= stepResult <= self.config.get('yMax')
            inXAxisRange = self.config.get('xMin') <= i <= self.config.get('xMax')

            if inYAxisRange and inXAxisRange:
                x.append(i)
                y.append(stepResult)
        
        return np.array([x, y])

    def addFunctionData(self, data:np.array):
        plt.plot(data[0], data[1])
    
    def showPlot(self):
        plt.show()


