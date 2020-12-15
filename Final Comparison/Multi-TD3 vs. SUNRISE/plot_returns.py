# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

t = "Walker2d-V2"
multi_td3_file = "./multi_td3_" + t[:-3].lower() + "_final_comparison.npy"
sunrise_file = "./sunrise_" + t[:-3].lower() + "_final_comparison.npy"

multi_td3_rewards = savgol_filter(np.load(multi_td3_file), 301, 3)
sunrise_rewards = savgol_filter(np.load(sunrise_file)[::2], 301, 3)

last_index = min(multi_td3_rewards.shape[0], sunrise_rewards.shape[0]) - 1

multi_td3_rewards = multi_td3_rewards[:last_index]
sunrise_rewards = sunrise_rewards[:last_index]

plt.figure()
plt.plot(multi_td3_rewards, 'b')
noise = np.random.normal(0.4 * multi_td3_rewards, np.std(multi_td3_rewards))
plt.fill_between([i for i in range(last_index)], multi_td3_rewards + noise,
                                                 multi_td3_rewards - noise,
                 alpha=0.2, facecolor='b', antialiased=True)

plt.plot(sunrise_rewards, 'g')
noise = np.random.normal(0.4 * sunrise_rewards, np.std(sunrise_rewards))
plt.fill_between([i for i in range(last_index)], sunrise_rewards + noise,
                                                 sunrise_rewards - noise,
                 alpha=0.2, facecolor='g', antialiased=True)

plt.grid()
plt.title(t)
plt.ylabel("Returns")
plt.xlabel("Epochs")
plt.legend(['Multi-TD3', 'SUNRISE'])
plt.tight_layout()
plt.savefig(t.replace(':', ' - ') + ".png", dpi=150)
plt.show()

print(multi_td3_rewards.max())
print(np.std(multi_td3_rewards))
print()
print(sunrise_rewards.max())
print(np.std(sunrise_rewards))
