import numpy as np
import matplotlib.pyplot as plt
import random

from icecream import ic


def func(x):
    return 10 * x**2 + 32*x + 9


def gradient(x):
    return 20 *x + 32


x = np.linspace(-10, 10)


steps = []

x_star = random.choice(x)

alpha = 1e-3

for i in range(100):
    x_star = x_star + -1*gradient(x_star)*alpha
    steps.append(x_star)

    ic(x_star, func(x_star))

fig, ax = plt.subplots()
ax.plot(x, func(x))

for i, s in enumerate(steps):
    ax.annotate(str(i+1), (s, func(s)))

plt.show()

