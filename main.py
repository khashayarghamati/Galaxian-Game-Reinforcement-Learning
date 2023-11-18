import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import (FrameStack,
                                GrayScaleObservation,
                                TransformObservation,
                                ResizeObservation)
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from agent import Agent
from metrics import MetricLogger

env = gym.make('ALE/Galaxian-v5',  render_mode='rgb_array')

# env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=6)



env.reset()



save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

vid = VideoRecorder(env=env, path="vid.mp4", base_path='/home2/kg23aay/')

checkpoint = None
agent = Agent(state_dim=6*84*84, action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 40000

for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:
        action = agent.act(state[0])

        next_state, reward, done, truncated, info = env.step(action)
        vid.capture_frame()
        q, loss = agent.learn(state=state[0], next_state=next_state, action=action, reward=reward)

        logger.log_step(reward, loss, q.cpu().detach().numpy())

        if done or truncated:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )
vid.close()