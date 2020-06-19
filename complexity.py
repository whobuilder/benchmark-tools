from enum import Enum
from scipy.optimize import curve_fit
import numpy as np


class Complexity:
    class BigO(Enum):
        o1 = 0
        oN = 1
        oN2 = 2
        oN3 = 3
        ologN = 4
        oNlogN = 5
        oAuto = 6
        default = o1

    def __init__(self, size, time, bigO):
        self.n = size
        self.time = time
        self.big_o = bigO
        self.coeff = None
        self.rms = None

    def calculate_complexity(self):
        if self.big_o == Complexity.BigO.oAuto:
            self.calculate_best_fit_complexity()
        else:
            self.coeff, self.rms = self.calculate_complexity_coefficient(
                Complexity.get_curve_fit_func(self.big_o)
            )

    def calculate_complexity_coefficient(self, func):
        coeff, _ = curve_fit(func, self.n, self.time)
        f_x = func(self.n, *coeff)
        rms = np.add.reduce((f_x - self.time) ** 2)
        mean = np.add.reduce(self.time) / np.size(self.n)
        rms = np.sqrt(rms / np.size(self.n)) / mean
        return coeff, rms

    def calculate_best_fit_complexity(self):
        self.big_o = Complexity.BigO.default
        self.coeff, self.rms = self.calculate_complexity_coefficient(
            Complexity.get_curve_fit_func(self.big_o)
        )
        for big_o in Complexity.BigO:
            if big_o == Complexity.BigO.default or big_o == Complexity.BigO.oAuto:
                continue  # default is already calculated and oAuto is being calculated now
            coeff, rms = self.calculate_complexity_coefficient(Complexity.get_curve_fit_func(big_o))
            if rms < self.rms:
                self.big_o = big_o
                self.rms = rms
                self.coeff = coeff

    @staticmethod
    # pylint: disable=invalid-name
    def o1(x, coeff):
        return coeff * np.ones(np.size(x))

    @staticmethod
    # pylint: disable=invalid-name
    def oN(x, coeff):
        return coeff * x

    @staticmethod
    # pylint: disable=invalid-name
    def oN2(x, coeff):
        return coeff * x ** 2

    @staticmethod
    # pylint: disable=invalid-name
    def oN3(x, coeff):
        return coeff * x ** 3

    @staticmethod
    # pylint: disable=invalid-name
    def ologN(x, coeff):
        return coeff * np.log(x)

    @staticmethod
    # pylint: disable=invalid-name
    def oNlogN(x, coeff):
        return coeff * x * np.log(x)

    @staticmethod
    def get_curve_fit_func(big_o):
        if big_o == Complexity.BigO.o1:
            return Complexity.o1
        elif big_o == Complexity.BigO.oN:
            return Complexity.oN
        elif big_o == Complexity.BigO.oN2:
            return Complexity.oN2
        elif big_o == Complexity.BigO.oN3:
            return Complexity.oN3
        elif big_o == Complexity.BigO.ologN:
            return Complexity.ologN
        elif big_o == Complexity.BigO.oNlogN:
            return Complexity.oNlogN
        else:
            raise IOError(f"Unknow complexity type ({big_o.name}) is given")
