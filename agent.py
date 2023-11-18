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

        self.discount_factor = 0.9

        self.curr_step = 0

        self.save_every = 22000
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        c, h, w = self.state_dim
        self.q_network = QNetwork(h, self.action_dim).cuda()
        if self.use_cuda:
            self.q_network = self.q_network.to(device='cuda')

        self.optimizer = optim.SGD(self.q_network.parameters(), lr=0.1)

        if checkpoint:
            self.load(checkpoint)

    def act(self, state):

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.q_network(state)
            print(f'action_values1 {action_values.shape}')
            print(f'action_values {action_values}')
            print(f'action_values axis1 {torch.argmax(action_values, axis=1)}')
            # print(f'action_value  {(torch.argmax(action_values, axis=0)[0][0]).item()}')


            # print(f'action_values axis4 {torch.argmax(action_values, axis=4)}')
            action_idx = (torch.argmax(action_values, axis=0)[0][0]).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def state_to_tensor(self, state):

        return torch.FloatTensor(state).cuda()

    def update_Q_online(self, state_tensor, action, reward, next_state_tensor):
        self.optimizer.zero_grad()

        state_tensor = self.state_to_tensor(state_tensor)

        q_values = self.q_network(state_tensor)

        q_value = q_values[action]

        next_state_tensor = self.state_to_tensor(next_state_tensor)

        target_q_value = reward + self.discount_factor * torch.max(self.q_network(next_state_tensor))
        loss = torch.nn.MSELoss()(q_value, target_q_value.detach())
        loss.backward()
        self.optimizer.step()
        return loss.item(), target_q_value

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
