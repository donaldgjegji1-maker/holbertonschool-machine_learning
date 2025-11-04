#!/usr/bin/env python3
"""A script that plots a stacked bar graph"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """A function that plots a stacked bar graph"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    people = ['Farrah', 'Fred', 'Felicia']
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]
    x = np.arange(len(people))

    plt.bar(x, apples, color='red', width=0.5, label='apples')
    plt.bar(
        x, bananas, bottom=apples,
        color='yellow', width=0.5, label='bananas'
    )
    plt.bar(
        x, oranges, bottom=apples + bananas,
        color='#ff8000', width=0.5, label='oranges'
    )
    plt.bar(
        x, peaches, bottom=apples + bananas + oranges,
        color='#ffe5b4', width=0.5, label='peaches'
    )

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(x, people)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.legend()
    plt.show()
