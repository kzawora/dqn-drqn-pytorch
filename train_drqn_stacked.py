from typing import Any
from random import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import gym
from tqdm import tqdm
from time import time
# custom classes
from env_wrappers import BreakoutEnv
from models import DRQN_shallow as DRQN
from replay_buffer import ReplayBuffer


@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    done: bool


def update_target_model(model, target):
    target.load_state_dict(model.state_dict())


def policy_evaluation(model, env, device, test_episodes=10, max_steps=1000):
    best_frames = []
    best_reward = -999999
    rewards = []
    eps_greedy = True
    eps = 0.05
    for _ in range(test_episodes):
        idx = 0
        done = False
        reward = 0
        same_frame_ctr = 0
        frames = []
        last_observation = env.reset()
        frames.append(env.frame)
        hidden = None
        while (not done) and (idx < max_steps):
            if eps_greedy and random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    x = torch.Tensor(last_observation).unsqueeze(0).to(device)
                    qvals, next_hidden = model(x, hidden)
                    action = qvals.max(-1)[-1].item()
                    hidden = next_hidden
            observation, r, done, _ = env.step(action)
            if np.array_equal(last_observation, observation):
                same_frame_ctr += 1
            else:
                same_frame_ctr = 0
            if same_frame_ctr > 200:
                print('TEST: No movement in over 200 frames! Aborting episode')
                same_frame_ctr = 0
                done = True
            last_observation = observation
            reward += r
            frames.append(env.frame)
        rewards.append(reward)
        if reward > best_reward:
            best_reward = reward
            best_frames = frames
    return np.mean(rewards), best_reward, np.stack(best_frames, 0)


def train_step(model, state_transitions, target, num_actions, device, gamma=0.99):
    curr_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions[:-1]]
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions[:-1]])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions[:-1]])).to(
        device)

    Q_batch, _ = model(curr_states[:-1])
    with torch.no_grad():
        next_Q_batch, _ = target(curr_states[1:])

    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(torch.float).to(device)
    loss_fn = torch.nn.SmoothL1Loss()
    Q_pred = torch.sum(Q_batch * one_hot_actions, -1)
    Q_tgt = rewards.squeeze() + mask[:, 0] * next_Q_batch.max(-1)[0] * gamma
    loss = loss_fn(Q_pred, Q_tgt)

    model.opt.zero_grad()
    loss.backward()
    model.opt.step()

    return loss


def main(test=False, checkpoint=None, device='cuda', project_name='drqn', run_name='example'):
    if not test:
        wandb.init(project=project_name, name=run_name)

    ## HYPERPARAMETERS
    memory_size = 500000
    min_rb_size = 100000
    sample_size = 32
    lr = 0.005
    boltzmann_exploration = False
    eps_min = 0.05
    eps_decay = 0.999995
    train_interval = 4
    update_interval = 10000
    test_interval = 5000
    episode_reward = 0
    episode_rewards = []
    screen_flicker_probability = 0.5

    # additional hparams
    living_reward = -0.01
    same_frame_ctr = 0
    same_frame_limit = 200

    # replay buffer
    replay = ReplayBuffer(memory_size, truncate_batch=True, guaranteed_size=32)
    step_num = -1 * min_rb_size

    # environment creation
    env = gym.make('BreakoutDeterministic-v4')
    env = BreakoutEnv(env, 84, 84)
    test_env = gym.make('BreakoutDeterministic-v4')
    test_env = BreakoutEnv(test_env, 84, 84)
    last_observation = env.reset()

    # model creation
    model = DRQN(env.observation_space.shape, env.action_space.n, lr=lr).to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    target = DRQN(env.observation_space.shape, env.action_space.n).to(device)
    update_target_model(model, target)

    # lstm hidden state
    hidden = None

    # training loop
    tq = tqdm()
    while True:
        if test:
            env.render()
            time.sleep(0.05)
        tq.update(1)
        eps = max(eps_min, eps_decay ** (step_num))
        if test:
            eps = 0
        if boltzmann_exploration:
            x = torch.Tensor(last_observation).unsqueeze(0).to(device)
            logits, next_hidden = model(x, hidden)
            action = torch.distributions.Categorical(logits=logits[0]).sample().item()
            hidden = next_hidden
        else:
            # epsilon-greedy
            if random() < eps:
                action = env.action_space.sample()
            else:
                x = torch.Tensor(last_observation).unsqueeze(0).to(device)
                qvals, next_hidden = model(x, hidden)
                action = qvals.max(-1)[-1].item()
                hidden = next_hidden

        # screen flickering
        # if random() < screen_flicker_probability:
        #    last_observation = np.zeros_like(last_observation)

        # observe and obtain reward
        observation, reward, done, info = env.step(action)
        episode_reward += reward

        # add to replay buffer
        replay.insert(Sarsd(last_observation, action, reward, done))
        last_observation = observation

        # episode end logic
        if done:
            hidden = None
            episode_rewards.append(episode_reward)
            if len(episode_rewards) > 100:
                del episode_rewards[0]
            wandb.log({
                "reward_ep": episode_reward,
                "avg_reward_100ep": np.mean(episode_rewards)
            })
            episode_reward = 0
            last_observation = env.reset()
        step_num += 1

        # testing, model updating and checkpointing
        if (not test) and (replay.idx > min_rb_size):
            if step_num % train_interval == 0:
                loss = train_step(model, replay.sample(sample_size), target, env.action_space.n, device)
                wandb.log(
                    {
                        "loss": loss.detach().cpu().item(),
                        "step": step_num
                    }
                )
                if not boltzmann_exploration:
                    wandb.log({"eps": eps})
            if step_num % update_interval == 0:
                print('updating target model')
                update_target_model(model, target)
                torch.save(target.state_dict(), f'target.model')
                model_artifact = wandb.Artifact("model_checkpoint", type="raw_data")
                model_artifact.add_file('target.model')
                wandb.log_artifact(model_artifact)
            if step_num % test_interval == 0:
                print('running test')
                avg_reward, best_reward, frames = policy_evaluation(model, test_env, device)  # model or target?
                wandb.log({'test_avg_reward': avg_reward,
                           'test_best_reward': best_reward,
                           'test_best_video': wandb.Video(frames.transpose(0, 3, 1, 2), str(best_reward), fps=24)})
    env.close()


if __name__ == "__main__":
    main(project_name='dqn_drqn_breakout_sandbox', run_name='[GTX970] drqn_shallow_32_guarantee_lr0_005')
