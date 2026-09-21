import numpy as np


class Response():
    def __init__(self, dim, length, time=None):
        self.disp = np.zeros((dim,length))
        self.vel = np.zeros((dim,length))
        self.acc = np.zeros((dim,length))
        self.resisting_force = None
        self.dim = dim
        self.length = length
        if time != None:
            self.interval = time.interval
            self.time = time.value
        self.aux = None
        self.element_disp = None
