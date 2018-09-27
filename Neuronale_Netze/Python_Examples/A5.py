import math

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

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

print('O1: ' + str(o1))
print('O2: ' + str(o2))
input1 = sigmoid(o1 * w11 + o2 * w21 + bias1)
input2 = sigmoid(o2 * w22 + o1 * w12 + bias2)
print(input1, input2)

print('Starting calculation...')
for x in range(1, 5):
    print('Step: ' + str(x))
    # results of step
    tmpResult1 = input1 * w11 + input2 * w21 + bias1
    tmpResult2 = input2 * w22 + input1 * w12 + bias2
    input1 = sigmoid(tmpResult1)
    input2 = sigmoid(tmpResult2)
    print(input1, input2)