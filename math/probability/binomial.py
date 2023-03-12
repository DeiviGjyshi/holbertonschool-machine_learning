#!/usr/bin/env python3
"""a class Binomial that represents a binomial distribution"""


class Binomial:
    """Binomial class constructor"""
    def __init__(self, data=None, n=1, p=0.5):
        self.n = int(round(n))
        self.p = float(p)
        if self.n <= 0:
            raise ValueError("n must be a positive value")
        if p <= 0 or p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            mean = float(sum(data) / n)
            variance = float(sum((x - mean) ** 2 for x in data) / (n - 1))
            self.n = int(round(mean ** 2 / (mean - variance)))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """PMF function"""
        if type(k) is not int:
            k = int(k)
        if (k < 0):
            return 0
        factorial_n = 1
        for i in range(self.n):
            factorial_n *= (i + 1)
        factorial_k = 1
        for j in range(k):
            factorial_k *= (j + 1)
        diff = self.n - k
        factorial_d = 1
        for g in range(diff):
            factorial_d *= (g + 1)
        coeff = factorial_n / (factorial_k * factorial_d)
        pmf = coeff * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        """CDF function"""
        if type(k) is not int:
            k = int(k)
        if (k < 0):
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
