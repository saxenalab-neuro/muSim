import numpy as np
import matplotlib.pyplot as plt

interp_fast = np.load('mouse_experiments/data/interp_fast.npy')
interp_slow = np.load('mouse_experiments/data/interp_slow.npy')
interp_1 = np.load('mouse_experiments/data/interp_1.npy')

plt.plot(interp_fast)
plt.show()

plt.plot(interp_slow)
plt.show()

plt.plot(interp_1)
plt.show()