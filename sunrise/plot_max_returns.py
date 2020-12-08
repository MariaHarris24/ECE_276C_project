# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("./results/returns_max_walker2d.npy")
plt.plot(rewards)
plt.plot(rewards[::2])
plt.plot(rewards[1::2])

