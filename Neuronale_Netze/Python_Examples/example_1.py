import math

class Example1(object):
    
    def __init__(self):
        self.data = []
        n = input("Maximal Number:")
        n = int(n) + 1
        for a in range(1, n):
            for b in range (a, n):
                c_square = a**2 + b**2
                c = int(math.sqrt(c_square))
                if ((c_square - c**2) == 0):
                    print(a, b, c)
