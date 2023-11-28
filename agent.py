import random, numpy as np
import torch
from pathlib import Path

from collections import deque

from Model import Model
from config import Config
from neptune_kit import Neptune
from q_network import QNetwork
import torch.optim as optim


class Agent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.exploration_rate = Config.exploration_rate
        self.exploration_rate_decay = Config.exploration_rate_decay
        self.exploration_rate_min = Config.exploration_rate_min

        self.discount_factor = Config.discount_factor

        self.curr_step = 0

        self.save_every = Config.save_every
        self.save_dir = save_dir

        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.q_network = Model(self.state_dim, self.action_dim).cuda()
            self.q_network = self.q_network.to(device='cuda')
        else:
            self.q_network = QNetwork(self.state_dim, self.action_dim)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=Config.lr)

        if checkpoint:
            self.load(checkpoint)

    def act(self, state):

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.tensor(state, dtype=torch.float, device='cuda').unsqueeze(0)
            self.q_value = self.q_network(state)
            action_idx = torch.argmax(self.q_value).item()

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

    def update_Q_online(self, state, action, reward, next_state_tensor, done):

        # next_state = self.to_tensor(np.reshape(next_state_tensor, [1, self.state_dim]))
        # state_tensor = self.to_tensor(np.reshape(state_tensor, [1, self.state_dim]))

        # Update Q-values using the Double Q-learning update rule

        state = torch.tensor(state, dtype=torch.float, device='cuda')
        next_state = torch.tensor(next_state_tensor, dtype=torch.float, device='cuda')

        q_evaluation = self.q_network(next_state)
        next_states_target_q_value = q_evaluation.gather(1, self.q_value.max(1)[1].unsqueeze(1)).squeeze(1)

        target = reward + self.discount_factor * next_states_target_q_value * (1 - done)

        self.q_value = self.q_network(state)

        selected_q_value = self.q_value.gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute the loss and perform a gradient descent step
        loss = torch.nn.MSELoss()(selected_q_value, q_evaluation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), target

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
            self.save()

        state, next_state, action, reward, done = self.recall()

        # Backpropagate loss through Q_online
        loss, q = self.update_Q_online(state=state, action=action, reward=reward,
                                       next_state_tensor=next_state, done=done)
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

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def sync_Q_target(self):
        self.q_network.target.load_state_dict(self.q_network.online.state_dict())
