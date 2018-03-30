from Python_Examples import example_1 as Examples
from Python_Examples import example_2 as Examples
from Python_Examples.EulerMethod import EulerMethod
import numpy as np

# the function to calculate
def y(x:float):
    return (x - (x * x * x))

# gets a function that will be used
eulerCalc = EulerMethod(y)

# get range of calculation
result = eulerCalc.getRealFunctionByRange(-10.0, 10.0)
eulerCalc.addFunctionData(result)
# multiply with -1 to get different example
result = [result[0] * -1 + 1.25, result[1]]
eulerCalc.addFunctionData(result)

eulerCalc.showPlot()