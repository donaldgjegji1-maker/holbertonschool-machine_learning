#!/usr/bin/env python3
"""A script that represents a Binomial distribution"""


class Binomial:
    """Class representing a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes a Binomial distribution.
        If data is provided, estimates n and p from the data.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p_est = 1 - (variance / mean)
            n_est = mean / p_est

            self.n = round(n_est)
            self.p = mean / self.n

    def factorial(self, x):
        """Computes factorial of x manually (no imports)."""
        if x == 0 or x == 1:
            return 1
        f = 1
        for i in range(2, x + 1):
            f *= i
        return f

    def pmf(self, k):
        """
        Calculates the Probability Mass Function (PMF)
        for a given number of successes k.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        # Combination: C(n, k) = n! / (k! * (n - k)!)
        comb = self.factorial(self.n) / (
            self.factorial(k) * self.factorial(self.n - k)
        )

        # PMF: C(n, k) * p^k * (1 - p)^(n - k)
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculate the cumulative distribution function (CDF)
        for a given number of successes k.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1

        total = 0
        for i in range(0, k + 1):
            total += self.pmf(i)
        return total
