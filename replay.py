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

env = gym.make('ALE/Galaxian-v5', render_mode='rgb_array')

# env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()


save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

vid = VideoRecorder(env=env, path="DQN29.mp4")

checkpoint = Path('agent_net_29.chkpt')
agent = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n,
              save_dir=save_dir, checkpoint=checkpoint, prefer_CPU=False)
agent.exploration_rate = agent.exploration_rate_min

logger = MetricLogger(save_dir, "Replay")

episodes = 400

for e in range(episodes):

    state = env.reset()
    if type(state) == tuple:
        state = state[0]
    while True:
        env.render()
        action = agent.act(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        agent.cache(state, next_state, action, reward, truncated)
        vid.capture_frame()

        is_done = truncated

        state = next_state

        if terminated or truncated:
            break
    if e % 20 == 0:
        logger.record_action_and_info(
            episode=e,
            step=agent.curr_step,
            action=action,
            info=info,
            reward=reward
        )

    # if is_done:
    #     print(f"Game is done episode: {e} step: {agent.curr_step}")
    #     break


vid.close()