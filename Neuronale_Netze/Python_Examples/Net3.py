import numpy
import scipy.special
import matplotlib.pyplot
'''
%matplotlib inline
'''
class Net3(object):

    def __init__(self, cInput = 3, cHidden = 4, cOutput = 1, eta = 0.1):
        self.iNodes = cInput
        self.hNodes = cHidden
        self.oNodes = cOutput

        self.wih = numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.who = numpy.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))

        self.eta = eta

        self.activation_function = lambda x: scipy.special.expit(4 * x)
        pass
    
    def train(self, inputs_list, targets_list):
        inputs  = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs  = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs  = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.eta * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.eta * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    
    def query(self, inputs_list):
        inputs  = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs  = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs  = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    def learn(self):
        testDataInCircle = self.createTestData(6000)
        testDataMixed    = self.createTestData(6000, -2.0, 2.0)
        testData = numpy.concatenate((testDataInCircle, testDataMixed), axis=0)
        #testData = self.createTestData(10000, -2.0, 2.0)
        for record in testData:
            target = record[2]
            record[2] = 1.0
            print(record)
            print(target)
            #[:-1]
            self.train(record, target)
    
    def createTestData(self, s = 10000, l = -1.0, h = 1.0, d = 1):
        # generate random data in range l & h with size s
        x = numpy.random.uniform(low=l, high=h, size=s)
        y = numpy.random.uniform(low=l, high=h, size=s)
        # round data on decimals d
        x = numpy.around(x, decimals=d)
        y = numpy.around(y, decimals=d)
        # create array with results
        z = numpy.zeros(s)
        # check whether point is in circle
        for n in range(0, s):
            result = x[n] ** 2 + y[n] ** 2
            if(result <= 1):
                z[n] = 0.8
        return numpy.c_[x, y, z]