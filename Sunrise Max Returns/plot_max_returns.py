# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

file = "./returns_max_halfcheetah_1102.npy"
t = "SUNRISE: HalfCheetah-V2"

rewards = np.load(file)
plt.figure()
plt.plot(rewards)
plt.grid()
plt.show()

plt.figure()
plt.plot(rewards[::2])
plt.grid()
plt.show()

plt.figure()
plt.plot(rewards[1::2])
plt.grid()
plt.title(t)
plt.ylabel("Returns")
plt.xlabel("Epochs")
plt.tight_layout()
plt.savefig(t.replace(':', ' - ') + ".png", dpi=150)
plt.show()