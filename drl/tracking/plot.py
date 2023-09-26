import numpy as np
import matplotlib.pyplot as plt

rewards = np.loadtxt('drl/tracking/episode_rewards')
policy_loss = np.loadtxt('drl/tracking/policy_loss')

plt.plot(rewards)
plt.xlabel("Timestep")
plt.ylabel("Rewards")
# Show the plot
plt.show()

plt.plot(policy_loss)
plt.xlabel("Timestep")
plt.ylabel("Loss")
# Show the plot
plt.show()