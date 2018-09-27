import sys
import time
import threading

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as Tk
from tkinter import messagebox

#from Python_Examples import example_1 as Examples
#from Python_Examples import example_2 as Examples
#from Python_Examples.EulerMethod import EulerMethod
#from Python_Examples.Net1 import Net1
#from Python_Examples.Net2 import Net2
#from Python_Examples.Net3 import Net3
from Python_Examples.NetCalculator import NetCalculator

sys.setrecursionlimit(1500)

# the function to calculate
def y(x:float):
    return (x - (x * x * x))

'''
# gets a function that will be used
eulerCalc = EulerMethod(y)

# get range of calculation
result = eulerCalc.getRealFunctionByRange(-2.0, 2.0)
eulerCalc.addFunctionData(result)
# multiply with -1 to get different example
#result = [result[0] * -1 + 1.25, result[1]]
#eulerCalc.addFunctionData(result)

eulerCalc.euler(-7, 0.001, 0.0001)
eulerCalc.addFunctionData(eulerCalc.eulerResult)
eulerCalc.euler(0, 0.001, 0.0001)
eulerCalc.addFunctionData(eulerCalc.eulerResult)
eulerCalc.euler(8, 0.001, 0.0001)
eulerCalc.addFunctionData(eulerCalc.eulerResult)
eulerCalc.showPlot()
'''

#net = Net2()

#net.learn()

#net.learnSpecific([0.7, 0.4, 0.8])

#print(net.activate([0.5, 0.5]))

#print(net.activate([-2, 5]))

#data = net.createTestData(1000, -2, 2)

exitFlag = 0

class App(Tk.Tk):
    def __init__(self, master, plotter):
        self.net = plotter
        self.f = Figure(dpi=150)

        self.ax1 = self.f.add_subplot(221)
        self.ax1.set_title("Lerndaten")
        self.ax1.set_xlabel("x-Achse")
        self.ax1.set_ylabel("y-Achse")
        self.ax1.set_xlim([-2.5, 2.5])
        self.ax1.set_ylim([-2.5, 2.5])
        im = self.ax1.scatter(self.net.testData[:, 0], self.net.testData[:, 1], c=self.net.testData[:, 2], s=0.5)
        self.f.colorbar(im)

        self.ax2 = self.f.add_subplot(222)
        self.ax2.set_title("Gewichte Input->Hidden")
        im = self.ax2.imshow(self.net.wInputHidden, interpolation='nearest', cmap=plt.cm.ocean, vmin=-1, vmax=1)
        self.f.colorbar(im)

        self.ax3 = self.f.add_subplot(223)
        self.ax3.set_title("Fehlerkurve")
        y = np.arange(0, len(self.net.displayErrorSeries), 1)
        x = self.net.displayErrorSeries
        self.ax3.plot(y, x)

        #self.ax3.set_title("Gewichte Hidden->Output")
        #im = self.ax3.imshow(np.matrix(self.net.wHiddenOutput), interpolation='nearest', cmap=plt.cm.ocean, vmin=-1, vmax=1)
        #self.f.colorbar(im)

        self.ax4 = self.f.add_subplot(224)
        self.ax4.set_title("Ergebniskurve")
        y = np.arange(0, len(self.net.displaySeries), 1)
        x = self.net.displaySeries
        self.ax4.plot(y, x)
        #im = self.ax4.imshow(self.net.outputSeries, interpolation='nearest', cmap=plt.cm.ocean, vmin=0, vmax=1)
        #self.f.colorbar(im)

        self.canvas = FigureCanvasTkAgg(self.f, master)

        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.toolbar = toolbar = NavigationToolbar2TkAgg(self.canvas, master)
        self.button = button = Tk.Button(master, text='Quit', command=self.btnExit)
        button.pack(side=Tk.BOTTOM)
        toolbar.update()
        self.canvas.draw()
    
    def btnExit(self):
        """When you click to exit, this function is called"""
        if True: #messagebox.askyesno("Exit", "Do you want to quit the application?"):
            print('Exit Application')
            plotter.exitFlag = 1
            exitFlag         = 1
            root.destroy()
    
    def updatePlots(self):
        self.updatePlotData()
        self.updatePlotWeight()
        self.updatePlotOutput()
    
    def updatePlotData(self):
        self.ax1.cla()
        self.ax1.set_xlim([-2.5, 2.5])
        self.ax1.set_ylim([-2.5, 2.5])
        self.ax1.scatter(self.net.testData[:, 0], self.net.testData[:, 1], c=self.net.testData[:, 2], s=0.5)
    
    def updatePlotWeight(self):
        self.ax2.cla()
        self.ax2.set_title("Weights Input-Hidden")
        self.ax2.imshow(self.net.wInputHidden, interpolation='nearest', cmap=plt.cm.ocean)
        #self.ax3.cla()
        #self.ax3.set_title("Weights Hidden-Output")
        #self.ax3.imshow(np.matrix(self.net.wHiddenOutput), interpolation='nearest', cmap=plt.cm.ocean)
    
    def updatePlotOutput(self):
        self.ax3.cla()
        self.ax3.set_title("Fehlerkurve")
        self.ax3.set_ylim((-1, 1))
        self.ax3.set_xlim((0, len(self.net.displayErrorSeries)))
        y = np.arange(0, len(self.net.displayErrorSeries), 1)
        x = self.net.displayErrorSeries
        self.ax3.plot(y, x, markersize=0.1)

        self.ax4.cla()
        self.ax4.set_title("Ergebniskurve")
        self.ax4.set_ylim((0, 1))
        self.ax4.set_xlim((0, 500))
        y = np.arange(0, len(self.net.displaySeries), 1)
        x = self.net.displaySeries
        self.ax4.plot(y, x, markersize=0.1)
        #self.ax4.imshow(, interpolation='nearest', cmap=plt.cm.ocean, vmin=0, vmax=1)
    
    def __call__(self):
        self.updatePlots()
        self.canvas.draw()

if __name__ == '__main__':
    plotter = NetCalculator()
    plotter.start()

    root = Tk.Tk()
    root.wm_title('Neuronal Network - Overview')
    app = App(root, plotter)

    def UpdatePlot():
        while not exitFlag:
            try:
                app()
                #print('Updating...')
                #time.sleep(0.1)
            except Exception as e:
                return
        # scheint zu funktionieren...
        root.destroy()
    
    def UpdateOutputPlot():
        while not exitFlag:
            try:
                # number as index for the plot to update
                app(2)
                #print('Updating')
            except Exception as e:
                return
        root.destroy
    
    t0 = threading.Thread(target=UpdatePlot)
    t0.start()
    t1 = threading.Thread(target=UpdateOutputPlot)
    t1.start()

    Tk.mainloop()

    # stopping thread
    plotter.exitFlag = 1

plotter.exitFlag = 1
exitFlag = 1


#n = Net3()

#n.learn()
#print('-----------------------------------------------------')
#print(n.query([0.12, 0.3, 1.0]))
#print(n.query([-1.6, 1.9, 1.0]))