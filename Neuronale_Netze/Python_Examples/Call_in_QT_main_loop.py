import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import threading
import time

from PyQt5 import QtCore

class Call_in_QT_main_loop(QtCore.QObject):
    signal = QtCore.pyqtSignal()

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.args = list()
        self.kwargs = dict()
        self.signal.connect(self._target)

    def _target(self):
        self.func(*self.args, **self.kwargs)

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.signal.emit()