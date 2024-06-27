# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from mswp import Minesweeper
import os

# %%

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 100)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# %%
def rewardFunction(done,numRevealed):
    if done:
        return -1
    else:
        reward = 0
        reward += numRevealed * .01
        return reward

    return 0


# %%
class Agent():
    def __init__(self, model):
        self.model = model
        self.target_model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.lossVal = 0
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def store (self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(100)

        with torch.no_grad():
            state = state.flatten()
            return torch.argmax(self.model(torch.tensor(state).float())).item()

    def train(self):
        for state,action,reward,next_state,done in self.memory:

            state = state.flatten()
            next_state = next_state.flatten()
            

            target = self.model(torch.tensor(state).float())
            target_next = self.target_model(torch.tensor(next_state).float())
            target_val = self.model(torch.tensor(state).float())

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * torch.max(target_next).item()

            self.optimizer.zero_grad()
            loss = self.loss(target_val, target)
            self.lossVal = loss.item()
            loss.backward()
            self.optimizer.step()

        
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.memory.clear()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# %%
agent = Agent(Model())
env = Minesweeper()
state = env.reset().astype(float)
done = False
ep=0

# %%

actions = 0
ep_count = 10000

while ep < ep_count:
    while not done:
        action = agent.act(state)
        y = action % 10
        x = action // 10
        done = env.reveal(x, y)
        actions += 1

        next_state = env.get_visible_grid().astype(float)
        # find number of revealed squares since last state
        numRevealed = 0
        for i in range(100):
            if next_state.flatten()[i] != -2:
                numRevealed += 1
        reward = rewardFunction(done,numRevealed)


        agent.store(state, action, reward, next_state, done)
        state = next_state
    if done:
        ep += 1
        done = False
        if actions > 10:
            env.print_grid()
        actions = 0
        state = env.reset().astype(float)
    if len(agent.memory) > 64:
        agent.train()
        print("Episode: ", ep, "Loss: ", agent.lossVal)

torch.save(agent.model.state_dict(), "model.pth")


