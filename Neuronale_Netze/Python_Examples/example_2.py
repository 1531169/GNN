'''
Created on 30.03.2018

@author: chibi
'''

import matplotlib.pyplot as plt

class Example2(object):
    '''
    classdocs
    '''


    def __init__(self):
        self.data = []
        plt.plot([1,2,3,4])
        plt.ylabel('x')
        plt.xlabel('t')
        plt.show()