import numpy as np
import matplotlib.pyplot as plt

data = np.load('../training_statistics/stats_rewards.npy')

plt.plot(data)
plt.title('Reward Graph')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.grid(False)
plt.tight_layout()

plt.savefig('../training_statistics/reward_plot.png')  # Saves into train_statistics folder, not Analysis
plt.close()