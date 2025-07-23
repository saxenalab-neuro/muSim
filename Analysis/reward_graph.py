import numpy as np
import matplotlib.pyplot as plt


def create_graph(load_path, save_path, env_name, window_size):
    data = np.load(load_path)

    # Compute moving average
    moving_avg = np.concatenate((np.zeros(window_size//2) * np.nan, np.convolve(data, np.ones(window_size)/window_size, mode='valid'), np.zeros(window_size//2) * np.nan))

    # Trendline (linear, degree=1)
    x = np.arange(0, len(data))
    coeffs = np.polyfit(x, data, deg=1)
    trendline = np.polyval(coeffs, x)

    plt.plot(data)
    plt.plot(moving_avg, label='Moving Average', color='red', linewidth=2)
    plt.plot(x, trendline, label='Trendline (Linear)', color='black', linestyle='--')
    plt.title(env_name + ' Reward Graph')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(save_path)  # Saves into train_statistics folder, not Analysis
    plt.close()

def create_curriculum_graph(env_names, window_size):
    data = np.load('../training_statistics/stats_rewards.npy')
    # Plotting
    fig, axes = plt.subplots(10, 1, figsize=(10, 20), sharex=True)

    # Find global min and max (ignoring NaNs)
    y_min = np.nanmin(data)
    y_max = np.nanmax(data)

    for i in range(10):
        y = data[:, i]
        x = np.arange(len(y))
        valid = ~np.isnan(y)

        axes[i].plot(x[valid], y[valid], linestyle='-', linewidth=2)  # No marker
        axes[i].set_ylabel(env_names[i])
        axes[i].grid(True)
        axes[i].set_ylim(y_min, y_max)  # Set same y-range for all

        # Moving average (only past values, causal)
        y_avg = np.full_like(y, np.nan, dtype=float)
        for j in range(len(y)):
            window = y[max(0, j - window_size + 1):j + 1]
            window = window[~np.isnan(window)]
            if len(window) > 0:
                y_avg[j] = np.mean(window)

        axes[i].plot(x, y_avg, color='red', linewidth=2, label='Moving Avg')

    plt.xlabel('Row Index')
    plt.tight_layout()
    plt.savefig('reward_plot.png')

envs = ['Reach', 'CurvedReachClk', 'CurvedReachCClk', 'Sinusoid', 'SinusoidInv', \
          'FullReach', 'CircleClk', 'CircleCClk', 'Figure8', 'Figure8Inv']

window_size = 100
create_curriculum_graph(envs, window_size)

#for env in envs:
    #create_graph('../training_statistics/' + env + '/stats_rewards.npy', './graphs/' + env + '/reward_plot.png', env)