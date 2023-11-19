import random, numpy as np
import torch
from pathlib import Path

from collections import deque
from neptune_kit import Neptune
from q_network import QNetwork
import torch.optim as optim


class Agent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1

        self.discount_factor = 0.99

        self.curr_step = 0

        self.save_every = 5e5
        self.save_dir = save_dir

        self.q_selection = None

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.q_network = QNetwork(self.state_dim, self.action_dim).cuda()
            self.q_network = self.q_network.to(device='cuda')
        else:
            self.q_network = QNetwork(self.state_dim, self.action_dim)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        if checkpoint:
            self.load(checkpoint)

    def act(self, state):

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = self.to_tensor(np.reshape(state, [1, self.state_dim]))
            self.q_selection, _ = self.q_network(state)
            action_idx = torch.argmax(self.q_selection).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def to_tensor(self, state):
        if self.use_cuda:
            return torch.FloatTensor(state).cuda()
        return torch.FloatTensor(state)

    def update_Q_online(self, state_tensor, action, reward, next_state_tensor):

        next_state = self.to_tensor(np.reshape(next_state_tensor, [1, self.state_dim]))
        state_tensor = self.to_tensor(np.reshape(state_tensor, [1, self.state_dim]))

        # Update Q-values using the Double Q-learning update rule
        _, q_evaluation = self.q_network(next_state)
        target = reward + self.discount_factor * q_evaluation[0][torch.argmax(self.q_selection).item()].item()
        q_selection, _ = self.q_network(state_tensor)
        q_selection[0][action] = target

        # Compute the loss and perform a gradient descent step
        loss = torch.nn.MSELoss()(q_selection, q_evaluation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), target

    def learn(self, state, next_state, action, reward):
        if self.curr_step % self.save_every == 0:
            self.save()

        # Backpropagate loss through Q_online
        loss, q = self.update_Q_online(state_tensor=state, action=action, reward=reward, next_state_tensor=next_state)
        return q, loss

    def save(self):
        save_path = self.save_dir / f"agent_net_{int(self.curr_step // self.save_every)}.chkpt"

        torch.save(
            dict(
                model=self.q_network.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"agent saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path.name, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.q_network.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
