import numpy as np
import matplotlib.pyplot as plt

data = np.load('../training_statistics/Figure8/stats_rewards.npy')


# Compute moving average
window_size = 5000
moving_avg = np.concatenate((np.zeros(window_size//2) * np.nan, np.convolve(data, np.ones(window_size)/window_size, mode='valid'), np.zeros(window_size//2) * np.nan))

# Trendline (linear, degree=1)
x = np.arange(0, len(data))
coeffs = np.polyfit(x, data, deg=1)
trendline = np.polyval(coeffs, x)

plt.plot(data)
plt.plot(moving_avg, label='Moving Average', color='red', linewidth=2)
plt.plot(x, trendline, label='Trendline (Linear)', color='black', linestyle='--')
plt.title('Reward Graph')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.grid(False)
plt.tight_layout()

plt.savefig('../training_statistics/reward_plot.png')  # Saves into train_statistics folder, not Analysis
plt.close()