import gym

import warmup  # noqa

print("Testing normal environments")
for env_str in ["humanreacher-v0"]:
    print(f"Testing {env_str}")
    env = gym.make(env_str)
    for ep in range(1):
        ep_steps = 0
        state = env.reset()
        while True:
            # repeat actions for some time to demonstrate muscle model
            if not ep_steps % 100:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            env.render()
            # We ignore environment terminations here for testing purposes
            if ep_steps >= 100000:
                break
            ep_steps += 1