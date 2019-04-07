import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


Experience = namedtuple("Experience", field_names="state action reward next_state done")


class BitFlipEnvironment:

    def __init__(self, bits):
        self.bits = bits
        self.state = torch.zeros((self.bits, ))
        self.goal = torch.zeros((self.bits, ))
        self.reset()

    def reset(self):
        self.state = torch.randint(2, size=(self.bits, ), dtype=torch.float)
        self.goal = torch.randint(2, size=(self.bits, ), dtype=torch.float)
        if torch.equal(self.state, self.goal):
            self.reset()
        return self.state.clone(), self.goal.clone()

    def step(self, action):
        self.state[action] = 1 - self.state[action]  # Flip the bit on position of the action
        reward, done = self.compute_reward(self.state, self.goal)
        return self.state.clone(), reward, done

    def render(self):
        print("State: {}".format(self.state.tolist()))
        print("Goal : {}\n".format(self.goal.tolist()))

    @staticmethod
    def compute_reward(state, goal):
        done = torch.equal(state, goal)
        return torch.tensor(0.0 if done else -1.0), done


class DuelingMLP(nn.Module):

    def __init__(self, state_size, num_actions):
        super().__init__()
        self.linear = nn.Linear(state_size, 256)
        self.value_head = nn.Linear(256, 1)
        self.advantage_head = nn.Linear(256, num_actions)

    def forward(self, x):
        x = x.unsqueeze(0) if len(x.size()) == 1 else x
        x = F.relu(self.linear(x))
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        action_values = (value + (advantage - advantage.mean(dim=1, keepdim=True))).squeeze()
        return action_values


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    """ Double-DQN with Dueling Architecture """

    def __init__(self, state_size, num_actions):

        self.state_size = state_size
        self.num_actions = num_actions

        self.gamma = 0.98
        self.batch_size = 128
        self.train_start = 1000

        self.memory = ReplayMemory(int(1e6))

        self.Q_network = DuelingMLP(state_size, num_actions)
        self.target_network = DuelingMLP(state_size, num_actions)
        self.update_target_network()

        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=0.001)

    def push_experience(self, state, action, reward, next_state, done):
        self.memory.push(Experience(state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def take_action(self, state, epsilon):
        if random.random() > epsilon:
            return self.greedy_action(state)
        else:
            return torch.randint(self.num_actions, size=())

    def greedy_action(self, state):
        with torch.no_grad():
            return self.Q_network(state).argmax()

    def optimize_model(self):
        if len(self.memory) < self.train_start:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        non_final_mask = ~torch.tensor(batch.done)
        non_final_next_states = torch.stack([s for done, s in zip(batch.done, batch.next_state) if not done])

        Q_values = self.Q_network(state_batch)[range(self.batch_size), action_batch]

        # Double DQN target #
        next_state_values = torch.zeros(self.batch_size)
        number_of_non_final = sum(non_final_mask)
        with torch.no_grad():
            argmax_actions = self.Q_network(non_final_next_states).argmax(1)
            next_state_values[non_final_mask] = self.target_network(non_final_next_states)[
                range(number_of_non_final), argmax_actions]

        Q_targets = reward_batch + self.gamma * next_state_values
        #####################

        assert Q_values.shape == Q_targets.shape

        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_values, Q_targets)
        loss.backward()
        self.optimizer.step()


def train(num_bits=10, num_epochs=10, hindsight_replay=True,
          eps_max=0.2, eps_min=0.0, exploration_fraction=0.5):

    """
    Training loop for the bit flip experiment introduced in https://arxiv.org/pdf/1707.01495.pdf using DQN or DQN with
    hindsight experience replay. Exploration is decayed linearly from eps_max to eps_min over a fraction of the total
    number of epochs according to the parameter exploration_fraction. Returns a list of the success rates over the
    epochs.
    """

    # Parameters taken from the paper, some additional once are found in the constructor of the DQNAgent class.
    future_k = 4
    num_cycles = 50
    num_episodes = 16
    num_opt_steps = 40

    env = BitFlipEnvironment(num_bits)

    num_actions = num_bits
    state_size = 2 * num_bits
    agent = DQNAgent(state_size, num_actions)

    success_rate = 0.0
    success_rates = []
    for epoch in range(num_epochs):

        # Decay epsilon linearly from eps_max to eps_min
        eps = max(eps_max - epoch * (eps_max - eps_min) / int(num_epochs * exploration_fraction), eps_min)
        print("Epoch: {}, exploration: {:.0f}%, success rate: {:.2f}".format(epoch + 1, 100 * eps, success_rate))

        successes = 0
        for cycle in range(num_cycles):

            for episode in range(num_episodes):

                # Run episode and cache trajectory
                episode_trajectory = []
                state, goal = env.reset()

                for step in range(num_bits):

                    state_ = torch.cat((state, goal))
                    action = agent.take_action(state_, eps)
                    next_state, reward, done = env.step(action.item())
                    episode_trajectory.append(Experience(state, action, reward, next_state, done))
                    state = next_state
                    if done:
                        successes += 1
                        break

                # Fill up replay memory
                steps_taken = step
                for t in range(steps_taken):

                    # Standard experience replay
                    state, action, reward, next_state, done = episode_trajectory[t]
                    state_, next_state_ = torch.cat((state, goal)), torch.cat((next_state, goal))
                    agent.push_experience(state_, action, reward, next_state_, done)

                    # Hindsight experience replay
                    if hindsight_replay:
                        for _ in range(future_k):
                            future = random.randint(t, steps_taken)  # index of future time step
                            new_goal = episode_trajectory[future].next_state  # take future next_state and set as goal
                            new_reward, new_done = env.compute_reward(next_state, new_goal)
                            state_, next_state_ = torch.cat((state, new_goal)), torch.cat((next_state, new_goal))
                            agent.push_experience(state_, action, new_reward, next_state_, new_done)

            # Optimize DQN
            for opt_step in range(num_opt_steps):
                agent.optimize_model()

            agent.update_target_network()

        success_rate = successes / (num_episodes * num_cycles)
        success_rates.append(success_rate)

    return success_rates


if __name__ == "__main__":
    bits = 50  # more than 10^15 states
    epochs = 40

    for her in [True, False]:
        success = train(bits, epochs, her)
        plt.plot(success, label="HER-DQN" if her else "DQN")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Success rate")
    plt.title("Number of bits: {}".format(bits))
    plt.savefig("{}_bits.png".format(bits), dpi=1000)
    plt.show()
