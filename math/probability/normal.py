#!/usr/bin/env python3
"""A script that implements a Normal distribution class"""


class Normal:
    """A class that represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """A function that initializes a Normal distribution instance"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            squared_differences = []
            for x in data:
                squared_differences.append((x - self.mean) ** 2)
            variance = sum(squared_differences) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        z = self.z_score(x)
        exponent = -0.5 * (z ** 2)
        coefficient = self.stddev * ((2 * pi) ** 0.5)
        pdf_value = (e ** exponent) / coefficient
        return pdf_value

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        pi = 3.1415926536
        z = self.z_score(x) / (2 ** 0.5)
        erf = (2 / (pi ** 0.5)) * (
            z - (z ** 3) / 3 + (z ** 5) / 10 - (z ** 7) / 42 + (z ** 9) / 216
            )
        cdf_value = 0.5 * (1 + erf)
        return cdf_value
