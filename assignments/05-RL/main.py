"""


"""
import sys

import gymnasium as gym
from customagent import Agent

import time

tic = time.perf_counter()

SHOW_ANIMATIONS = True

env = gym.make("LunarLander-v2", render_mode="human" if SHOW_ANIMATIONS else "none")
observation, info = env.reset(seed=42)

agent = Agent(
    action_space=env.action_space,
    observation_space=env.observation_space,
)

total_reward = 0
last_n_rewards = []
j = 0
for i in range(100000):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    agent.learn(observation, reward, terminated, truncated)
    total_reward += reward

    if terminated or truncated:
        j += 1
        observation, info = env.reset()
        last_n_rewards.append(total_reward)
        n = min(30, len(last_n_rewards))
        avg = sum(last_n_rewards[-n:]) / n
        improvement_emoji = "🔥" if (total_reward > avg) else "😢"
        print(
            f"{improvement_emoji} {j} Finished with reward {int(total_reward)}.\tAverage of last {n}: {int(avg)}"
        )
        if avg > 0:
            print(i)
            print("🎉 Nice work! You're ready to submit the leaderboard! 🎉")
            toc = time.perf_counter()
            print(f"time: {toc - tic:0.4f} seconds.")
            sys.exit(0)
        total_reward = 0

env.close()
