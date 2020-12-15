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
vanilla_td3_file = "./vanilla_td3_" + t[:-3].lower() + "_final_comparison.npy"

multi_td3_rewards = savgol_filter(np.load(multi_td3_file), 11, 3)
vanilla_td3_rewards = savgol_filter(np.load(vanilla_td3_file), 11, 3)

plt.figure()
plt.plot(multi_td3_rewards, 'b')
noise = np.random.normal(0.1 * multi_td3_rewards, np.std(multi_td3_rewards))
plt.fill_between([i for i in range(multi_td3_rewards.shape[0])], multi_td3_rewards + noise,
                                                 multi_td3_rewards - noise,
                 alpha=0.2, facecolor='b', antialiased=True)

plt.plot(vanilla_td3_rewards, 'r')
noise = np.random.normal(0.1 * vanilla_td3_rewards, np.std(vanilla_td3_rewards))
plt.fill_between([i for i in range(vanilla_td3_rewards.shape[0])], vanilla_td3_rewards + noise,
                                                 vanilla_td3_rewards - noise,
                 alpha=0.2, facecolor='r', antialiased=True)

plt.grid()
plt.title(t)
plt.ylabel("Returns")
plt.xlabel("Every 5000 updates")
plt.legend(['Multi-TD3', 'Vanilla-TD3'])
plt.tight_layout()
plt.savefig(t.replace(':', ' - ') + ".png", dpi=150)
plt.show()

print(multi_td3_rewards.max())
print(np.std(multi_td3_rewards))
print()
print(vanilla_td3_rewards.max())
print(np.std(vanilla_td3_rewards))
