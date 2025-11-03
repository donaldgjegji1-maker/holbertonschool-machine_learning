#!/usr/bin/env python3
"""A script that plots all 5 previous graphs in one figure"""

import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """A function that plots all 5 previous graphs in one figure"""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('All in One', fontsize='x-small')

    # Plot 1: Line graph
    plt.subplot(3, 2, 1)
    plt.plot(y0, color='red')
    plt.xlim(0, 10)

    # Plot 2: Scatter plot
    plt.subplot(3, 2, 2)
    plt.scatter(x1, y1, c='magenta')
    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    # Plot 3: Exponential decay
    plt.subplot(3, 2, 3)
    plt.plot(x2, y2)
    plt.yscale('log')

    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.xlim(x2[0], x2[-1])

    # Plot 4: Two decay curves
    plt.subplot(3, 2, 4)
    plt.plot(x3, y31, c='r', linestyle='dashed', label='C-14')
    plt.plot(x3, y32, c='g', label='Ra-226')
    plt.title("Exponential Decay of Radioactive Elements")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend()

    # Plot 5: Histogram (spans two columns)
    plt.subplot(3, 1, 3)
    plt.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')

    plt.title("Project A")
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, step=10))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
