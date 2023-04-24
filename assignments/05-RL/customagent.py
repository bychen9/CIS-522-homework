import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import NamedTuple

GAMMA = 0.99  # discount factor
TAU = 1e-3
EPSILON_START = 1
DECAY_RATE = 0.9995
EPSILON_END = 0.1
BATCH_SIZE = 32
LR = 1e-5
UPDATE_EVERY = 1
BUFFER_SIZE = 100000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 522


class Agent:
    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        random.seed(seed)
        # self.action_space = action_space
        # self.observation_space = observation_space
        # self.action_size = action_space.n
        self.state_size = observation_space.shape[0]
        # # self.qnet = QNetwork(self.state_size, self.action_size, seed).to(device)
        # self.qnet = QNetwork(self.state_size, self.action_size).to(device)

        # # self.target_net = QNetwork(self.state_size, self.action_size, seed).to(device)
        # self.optimizer = optim.Adam(self.qnet.parameters(), lr=LR)
        # self.eps = EPSILON_START

        self.n_actions = action_space.n
        self.replay_buffer = ReplayBuffer(self.state_size, 1, BUFFER_SIZE)

        self.epsilon = EPSILON_START
        self.q_net = QNetwork(self.state_size, self.n_actions).to(device)
        self.prev_obs = None
        self.prev_action = None
        self.prev_reward = None

        self.policy = self.epsilon_greedy(action_space.n)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)

    def epsilon_greedy(self, state):
        def policy_fn(q_net, state):
            if torch.rand(1) < self.epsilon:
                return torch.randint(self.n_actions, size=(1,), device=device).item()
            else:
                with torch.no_grad():
                    q_pred = q_net(state)
                    return torch.argmax(q_pred).view(1).item()

        return policy_fn(self.q_net, state)

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        # # self.qnet.eval()
        # self.eps = max(EPSILON_END, self.eps * DECAY_RATE)
        # if torch.rand(1) < self.eps:
        #     action = np.random.choice(self.action_size)
        # else:
        #     with torch.no_grad():
        #         q_pred = self.qnet(torch.Tensor(observation))
        #     action = np.argmax(q_pred.detach().numpy())
        # # self.qnet.train()
        # return action
        with torch.no_grad():
            self.epsilon = max(EPSILON_END, self.epsilon * DECAY_RATE)
            # print(self.epsilon)
            return self.epsilon_greedy(observation)

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if terminated or truncated:
            self.replay_buffer.add(
                self.prev_obs, self.prev_action, self.prev_reward, GAMMA, observation
            )
            return

        if self.prev_obs is None:
            self.prev_obs = observation

        action = self.act(observation)

        if self.prev_action is None:
            self.prev_action = action

        if self.prev_reward is None:
            self.prev_reward = reward

        self.replay_buffer.add(
            self.prev_obs, self.prev_action, self.prev_reward, GAMMA, observation
        )

        batch = self.replay_buffer.sample(BATCH_SIZE)
        q_pred = self.q_net(batch.state).gather(1, batch.action)

        self.prev_obs = observation
        self.prev_action = action
        self.prev_reward = reward

        with torch.no_grad():
            q_next_actions = self.q_net(batch.next_state)
            q_target = q_next_actions.max(dim=1)[0].view(-1, 1)

        # Compute the MSE loss between the predicted and target values
        loss = F.mse_loss(q_pred, q_target)

        # backpropogation to update the q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # def learn(
    #         self,
    #         observation: gym.spaces.Box,
    #         reward: float,
    #         terminated: bool,
    #         truncated: bool,
    # ) -> None:
    #     if terminated or truncated:
    #         self.replay_buffer.add(self.prev_obs, self.prev_action, self.prev_reward, GAMMA, observation)
    #         print(self.epsilon)
    #         return
    #     # action = self.act(state)
    #     # batch = self.replay_buffer.sample(BATCH_SIZE)
    #     # q_actions = self.qnet(batch.state)
    #     # q_pred = q_actions.gather(1, batch.action)
    #     # with torch.no_grad():
    #     #     q_next_actions = self.qnet(batch.next_state)
    #     #     max_acts = q_next_actions.argmax(dim=1).view(-1, 1)
    #     #     q_target_actions = self.target_net(batch.next_state)
    #     #     q_target = q_target_actions.gather(1, max_acts)
    #     #     # q_target = q_target_actions.max(dim=1)[0].view(-1, 1)
    #     #     q_target = batch.reward + q_target * batch.discount
    #     # loss = F.mse_loss(q_pred, q_target)

    #     # self.optimizer.zero_grad()
    #     # loss.backward()
    #     # self.optimizer.step()

    #     # soft_update_from_to(self.qnet, self.target_net)
    #     # Compute our predicted q-value given the state and action from our batch
    #     action = torch.argmax(self.q_net(observation)).view(1).item()
    #     # print("OBSERVATION HERE: ", observation)
    #     # print("END OF OBSERVTION")

    #     # action = self.act(observation)
    #     # print("HERE 2: ", self.act(observation))

    #     if self.prev_obs is None:
    #         self.prev_obs = observation

    #     if self.prev_action is None:
    #         self.prev_action = action

    #     if self.prev_reward is None:
    #         self.prev_reward = reward

    #     self.replay_buffer.add(self.prev_obs, self.prev_action, self.prev_reward, GAMMA, observation)

    #     batch = self.replay_buffer.sample(BATCH_SIZE)
    #     q_actions = self.q_net(batch.state)
    #     q_pred = q_actions.gather(1, batch.action)

    #     # q_pred = self.q_net(self.prev_obs).gather(0, torch.tensor(action).type(torch.LongTensor))

    #     self.prev_obs = observation
    #     self.prev_action = action
    #     self.prev_reward = reward

    #     # q_pred = torch.argmax(self.q_net(observation))

    #     # Now compute the q-value target (also called td target or bellman backup)
    #     # we don't need to compute gradients on the q-value target, just the q-value
    #     # prediction, so we disable autograd here to speed up performance
    #     # print(q_pred)
    #     # with torch.no_grad():
    #     #     # q_next_observations = self.q_net(batch.next_state)
    #     #     # max_obs = q_next_observations.argmax(dim=1).view(-1, 1)
    #     #     # print(max_obs)
    #     #     # print(observation)

    #     #     # First get the best q-value from the next state
    #     #     # q_target = self.q_net(observation).max(dim=0)[0].view(-1, 1)
    #     #     q_target = self.q_net(observation).max(dim=0)[0].view(-1, 1)

    #     #     # print(self.q_net(observation).max(dim=0))
    #     #     # Next apply the reward and discount to get the q-value target
    #     #     q_target = reward + GAMMA * q_target

    #     with torch.no_grad():
    #         # q_next_actions = self.q_net(batch.next_state)
    #         # max_acts = q_next_actions.argmax(dim=1).view(-1, 1)
    #         q_target_actions = self.q_net(batch.next_state)
    #         # q_target = q_target_actions.gather(1, max_acts)
    #         q_target = q_target_actions.max(dim=1)[0].view(-1, 1)
    #         # q_target = q_target_actions.max(dim=1)[0].view(-1, 1)
    #         q_target = batch.reward + q_target * batch.discount

    #     # Compute the MSE loss between the predicted and target values
    #     loss = F.mse_loss(q_pred, q_target)

    #     # backpropogation to update the q network
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # self.conv = nn.Conv2d(
        #     in_channels=n_channels, out_channels=16, kernel_size=3, stride=1
        # )
        self.fc0 = nn.Linear(in_features=state_size, out_features=128)
        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=action_size)

    def forward(self, x):
        ####################################################################
        # Fill in missing code below (...),
        # then remove or comment the line below to test your function
        # raise NotImplementedError("Q network")
        ####################################################################

        # Pass the input through the convnet layer with ReLU activation
        # x = F.relu(self.conv(x))
        # Flatten the result while preserving the batch dimension
        # x = torch.flatten(x, start_dim = 1)
        # Pass the result through the first linear layer with ReLU activation
        x = torch.tensor(x).type(torch.FloatTensor)
        x = F.relu(self.fc0(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # Finally pass the result through the second linear layer and return
        x = self.fc3(x)
        return x


# class QNetWork(nn.Module):
#     def __init__(self, state_size, action_size, seed):
#         super(QNetWork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(in_features=state_size, out_features=128)
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(in_features=128, out_features=128)
#         self.fc4 = nn.Linear(in_features=128, out_features=64)
#         self.fc3 = nn.Linear(in_features=64, out_features=action_size)

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc3(x)
#         return x


class Batch(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_state: torch.Tensor


class ReplayBuffer:
    def __init__(self, state_dim, act_dim, buffer_size):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.n_samples = 0

        self.state = torch.zeros(
            buffer_size, state_dim, dtype=torch.float32, device=device
        )
        self.action = torch.zeros(
            buffer_size, act_dim, dtype=torch.int64, device=device
        )
        self.reward = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)
        self.discount = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)
        self.next_state = torch.zeros(
            buffer_size, state_dim, dtype=torch.float32, device=device
        )

    def add(self, state, action, reward, discount, next_state):
        self.state[self.ptr] = torch.Tensor(state)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.discount[self.ptr] = discount
        self.next_state[self.ptr] = torch.Tensor(next_state)

        if self.n_samples < self.buffer_size:
            self.n_samples += 1

        self.ptr = (self.ptr + 1) % UPDATE_EVERY

    def sample(self, batch_size):
        idx = np.random.choice(self.n_samples, batch_size)
        state = self.state[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        discount = self.discount[idx]
        next_state = self.next_state[idx]

        return Batch(state, action, reward, discount, next_state)


# def soft_update_from_to(source, target):
#     for target_param, param in zip(target.parameters(), source.parameters()):
#         target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
