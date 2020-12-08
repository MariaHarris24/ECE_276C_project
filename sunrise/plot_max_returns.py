# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("./returns_max_walker2d.npy")
plt.figure()
plt.plot(rewards)
plt.show()

plt.figure()
plt.plot(rewards[::2])
plt.show()

plt.figure()
plt.plot(rewards[1::2])
plt.show()

