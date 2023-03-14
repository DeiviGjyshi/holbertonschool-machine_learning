#!/usr/bin/env python3
"""a class Normal that represents a normal distribution"""


class Normal:
    """class normal constructor"""
    def __init__(self, data=None, mean=0., stddev=1.):
        self.mean = float(mean)
        self.stddev = float(stddev)
        sum_of_squares = 0
        if self.stddev <= 0:
            raise ValueError("stddev must be a positive value")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data)) / len(data)
            for i in data:
                diff = i - self.mean
                sum_of_squares += diff ** 2
            self.stddev = (sum_of_squares / len(data)) ** 0.5

    def z_score(self, x):
        """Z-score function"""
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """X value function"""
        x = (z * self.stddev) + self.mean
        return x

    def pdf(self, x):
        """PDF function"""
        pi = 3.1415926536
        e = 2.7182818285
        coeff = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -0.5 * (((x - self.mean) / self.stddev) ** 2)
        return coeff * (e ** exponent)

    def cdf(self, x):
        """CDF function"""
        pi = 3.1415926536
        v = float((x - self.mean) / (self.stddev * (2 ** 0.5)))
        va2 = (v ** 3 / 3) + (v ** 5 / 10) - (v ** 7 / 42) + (v ** 9 / 216)
        coeff = (v - va2 )
        erf = float((2 / (pi ** 0.5)) * coeff)
        cdf = float((1/2) * (1 + erf))
        return cdf
