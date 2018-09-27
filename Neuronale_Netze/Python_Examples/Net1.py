'''
Created on 30.04.2018

@author: chibi
'''
import numpy as np
import matplotlib.pyplot as plt

class Net1():
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    
    def getTestData(self, l, h, s, d):
        # generate random data in range l & h with size s
        x = np.random.uniform(low=l, high=h, size=s)
        y = np.random.uniform(low=l, high=h, size=s)
        # round data on decimals d
        x = np.around(x, decimals=d)
        y = np.around(y, decimals=d)
        return np.c_[x, y]
    
    def showData(self):
        data = self.getTestData(-2, 2, 100, 3)
        x = data[0]
        y = data[1]
        #plt.axis([-300, 300, -300, 300])
        n = 1024
        X = np.random.uniform(low=-2, high=2, size=n)
        Y = np.random.uniform(low=-2, high=2, size=n)

        plt.imshow(data, cmap=plt.cm.gray)
        #plt.scatter(X,Y)
        plt.show()