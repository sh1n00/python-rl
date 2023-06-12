import random
from dataclasses import dataclass

from enviroment import Environment


@dataclass
class Agent:
    def __init__(self, env):
        self.actions = env.actions

    def policy(self):
        return random.choice(self.actions)


def main():
    # Make grid environment.
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    # Try 10 game.
    for i in range(1, 11):
        # Initialize position of agent.
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy()
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Episode {i}: Agent gets {total_reward} reward.")


if __name__ == "__main__":
    main()
