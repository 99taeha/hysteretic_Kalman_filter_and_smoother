import numpy as np


class Time():
    def __init__(self, start, end, interval):
        self.start = start
        self.end = end
        self.interval = interval
        self.value = np.arange(start, end+1e-10, interval)
        self.length = len(self.value)
