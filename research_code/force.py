import numpy as np

from scipy.interpolate import interp1d


class Force():
    def interp(self, time):
        if time.end > self.time_point[-1]:
            raise ValueError('The time point is not enough to interpolate the force')
        f_interp = interp1d(self.time_point, self.value, kind='linear')
        self.time_point = time.value
        self.value = f_interp(self.time_point)

    @classmethod
    def assign_by_value(cls, time_point, value):
        instance = cls()
        value = np.atleast_2d(value)
        instance.time_point = time_point
        instance.number = value.shape[0]
        instance.value = value
        if value.shape[1] != len(time_point):
            raise ValueError('The size of time_point and value is not matched')
        return instance
