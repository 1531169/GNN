'''
Created on 30.03.2018

@author: chibi
'''
from array import array
import matplotlib.pyplot as plt

class EulerMethod(object):
    '''
    classdocs
    '''


    def __init__(self, deltaT):
        self.deltaT = deltaT
    
    def explicitMethod(self, initialConditions:array, deltaT):
        return 0
    
    def getNextStep(self, curStep = 0):
        return curStep + self.deltaT
    
    def getRealFunctionByRange(self, xFrom, xTo):
        return 0
    
    def showPlot(self):
        plt.plot([0.1, 0.2, 0.3], [-7, -0.2, 8])
        plt.ylabel('x')
        plt.xlabel('t')
        plt.show()


