import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import (FrameStack,
                                GrayScaleObservation,
                                TransformObservation,
                                ResizeObservation)
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from agent import Agent
from config import Config
from metrics import MetricLogger

env = gym.make('ALE/Galaxian-v5',  render_mode='rgb_array')

# env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)



env.reset()



save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

# vid = VideoRecorder(env=env, path="vid.mp4")

checkpoint = None
agent = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = Config.total_episode

for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:
        action = agent.act(state[0])

        next_state, reward, terminated, truncated, info = env.step(action)

        agent.cache(state[0], next_state, action, reward, terminated)

        # vid.capture_frame()
        q, loss = agent.learn()

        logger.log_step(reward, loss, q, info)
        is_done = terminated
        if terminated or truncated:
            break
    logger.log_episode()

    if is_done:
        print(f"Game is done episode: {e} step: {agent.curr_step}")
        agent.save()
        break

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )
# vid.close()