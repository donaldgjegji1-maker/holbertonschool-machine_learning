#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""A script that plots a histogram of student scores for a project"""


def frequency():

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')

    plt.title("Project A")
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, step=10))
    plt.show()
