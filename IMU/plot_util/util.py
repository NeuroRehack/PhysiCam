import sys
import math
import numpy as np
from scipy import signal as sig


class Util:

    bounds = (0, 1)  # upper and lower bounds used for normalisation
    filt_order = 1

    def derive(time, data):
        """
        returns the time-series derivative

        calculates derivative of the current point as follows:
        1) finds the gradient between the current point and the previous point
        2) finds the gradient between the current point and the next point
        3) takes the average of the two gradients and sets as gradient of current point

        """
        derivative = list()
        data_len = len(data)

        for i, d in enumerate(data):
            if i == 0 or i == data_len - 1:
                derivative.append(None)
            elif data[i - 1] is not None and data[i] is not None and data[i + 1] is not None:
                d_prev = (data[i] - data[i - 1]) / (time[i] - time[i - 1])
                d_next = (data[i + 1] - data[i]) / (time[i + 1] - time[i])
                derivative.append((d_prev + d_next) / 2)
            else:
                derivative.append(None)

        return derivative.copy()


    def get_magnitude_2d(x_mag, y_mag):
        """
        returns the magnitude of 2-dim co-ord values (x, y)

        """
        mag = list()

        for x, y in zip(x_mag, y_mag):
            if x is not None and y is not None:
                mag.append(math.sqrt(x**2 + y**2))
            else:
                mag.append(None)

        return mag.copy()


    def get_magnitude_3d(x_mag, y_mag, z_mag):
        """
        returns the magnitude of 3-dim co-ord values (x, y, z)

        """
        mag = list()

        for x, y, z in zip(x_mag, y_mag, z_mag):
            if x is not None and y is not None:
                mag.append(math.sqrt(x**2 + y**2 + z**2))
            else:
                mag.append(None)

        return mag.copy()


    def remove_gravity(data, unit):
        """
        removes gravity component
        to be used on the wearable sensor data

        """
        new_data = list()

        for d in data:
            new_data.append(abs(d - unit))

        return new_data.copy()


    def normalise(data, bounds):
        """
        normalise data within specified bounds

        """
        lower, upper = bounds

        def get_max():
            max = -sys.maxsize
            for d in data:
                if d is not None and d > max:
                    max = d
            return max

        def get_min():
            min = sys.maxsize
            for d in data:
                if d is not None and d < min:
                    min = d
            return min

        norm = list()
        min, max = get_min(), get_max()

        for d in data:
            if d is not None:
                norm.append((d - min) / (max - min) * (upper - lower) + lower)
            else:
                norm.append(None)

        return norm.copy()


    def remove_none(time, data, new_time, new_data):
        """
        removes all "None" objects from the data list
        removes corresponding timestamp in the time list

        """
        for t, d in zip(time, data):
            if d is not None:
                new_time.append(t)
                new_data.append(d)


    def low_pass_filter(data, fc, fs, order):
        """
        butterworth low pass filter

        """
        w = fc / (fs / 2)
        b, a = sig.butter(order, w)
        out = sig.filtfilt(b, a, np.array(data))

        return out.copy()
