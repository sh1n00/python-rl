import random
import numpy as np
from dataclasses import dataclass


@dataclass
class CoinToss:
    head_probs: list
    max_episode_steps: int = 30
    toss_count: int = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception("The step count exceeded maximum. Please reset env.")
        else:
            done = True if self.toss_count == final else False

        if action >= len(self.head_probs):
            raise Exception(f"The No. {action} coin doesn't exist.")
        else:
            head_prob = self.head_probs[action]
            reward = 1.0 if random.random() < head_prob else 0.0
            self.toss_count += 1
            return reward, done


@dataclass
class EpsilonGreedyAgent:
    epsilon: float
    V: list = list

    def policy(self):
        coins = range(len(self.V))
        if random.random() < self.epsilon:
            return random.choice(coins)
        else:
            return np.argmax(self.V)

    def print(self):
        for v in self.V:
            print(v)

    def play(self, env):
        # Initialize estimation
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average

        return rewards


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt


    def main():
        env = CoinToss([0.1, 0.5, 0.9])
        epsilon = [0.0, 0.4, 0.9]
        game_steps = list(range(10, 310, 10))
        result = {}
        for e in epsilon:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = 5
                rewards = agent.play(env)
                means.append(np.mean(rewards))
            result[f"epsilon={e}"] = means
        result["coin toss count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("coin toss count", drop=True, inplace=True)

        result.plot.line()
        agent.print()
        plt.show()


    main()
