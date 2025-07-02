import numpy as np
import matplotlib.pyplot as plt

# Load the 1D NumPy array
data = np.load('../training_statistics/stats_rewards.npy')  # Replace with your actual filename

# Plot the data
plt.plot(data)
plt.title('1D Data Plot')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.grid(False)
plt.tight_layout()

# Save the figure instead of displaying it
plt.savefig('../training_statistics/reward_plot.png')  # You can change the filename and format (e.g., .pdf, .svg)
plt.close()