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
from replay_buffer import ReplayBuffer
from models import DRQN_shallow as DRQN


@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
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
            with torch.no_grad():
                x = torch.Tensor(last_observation).unsqueeze(0).to(device)
                qvals, hidden = model(x, hidden)
            if eps_greedy and random() < eps:
                action = env.action_space.sample()
            else:
                action = qvals.max(-1)[-1].item()
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


def train_step(model, state_transitions, target, num_actions, lr, device, gamma=0.99):
    curr_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]
    with torch.no_grad():
        qvals_next, _ = target(next_states)
        qvals_next = qvals_next.max(-1)[0]

    model.opt.zero_grad()
    for g in model.opt.param_groups:
        g['lr'] = lr

    qvals, _ = model(curr_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss_fn = torch.nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals * one_hot_actions, -1), rewards.squeeze() + mask[:, 0] * qvals_next * gamma)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 10)
    model.opt.step()
    return loss


def main(test=False, checkpoint=None, device='cuda', project_name='drqn', run_name='example'):
    if not test:
        wandb.init(project=project_name, name=run_name)

    ## HYPERPARAMETERS
    memory_size = 500000  # DARQN paper - 500k, 400k DRQN paper
    min_rb_size = 50000  # ? z dupy wyciagniete, po ilu iteracjach zaczynamy trening
    sample_size = 32  # ? z dupy, 32 DARQN
    lr = 0.001  # DARQN paper - 0.01
    lr_min = 0.00025
    lr_decay = (lr - lr_min) / 1e6
    boltzmann_exploration = False  # nie bylo w papierach
    eps = 1
    eps_min = 0.1  # DARQN - 0.1
    eps_decay = (eps - eps_min) / 1e6  # powinien byc liniowy
    train_interval = 4  # DARQN - 4
    update_interval = 10000  # wszystkie papiery
    test_interval = 5000  # z dupy, bez znaczenia do zbieznosci
    episode_reward = 0
    episode_rewards = []
    screen_flicker_probability = 0.5

    # replay buffer
    replay = ReplayBuffer(memory_size, truncate_batch=True, guaranteed_size=6)
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

    hidden = None
    # training loop
    tq = tqdm()
    while True:
        if test:
            env.render()
            time.sleep(0.05)
        tq.update(1)
        eps = max(eps_min, eps - eps_decay)
        lr = max(lr_min, lr - lr_decay)
        if test:
            eps = 0
        if boltzmann_exploration:
            x = torch.Tensor(last_observation).unsqueeze(0).to(device)
            logits, hidden = model(x, hidden)
            action = torch.distributions.Categorical(logits=logits[0]).sample().item()
        else:
            # epsilon-greedy
            x = torch.Tensor(last_observation).unsqueeze(0).to(device)
            qvals, hidden = model(x, hidden)
            if random() < eps:
                action = env.action_space.sample()
            else:
                action = qvals.max(-1)[-1].item()

        # screen flickering
        # if random() < screen_flicker_probability:
        #    last_observation = np.zeros_like(last_observation)

        # observe and obtain reward
        observation, reward, done, info = env.step(action)
        episode_reward += reward

        # add to replay buffer
        replay.insert(Sarsd(last_observation, action, reward, observation, done))
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
                loss = train_step(model, replay.sample(sample_size), target, env.action_space.n, lr, device)
                wandb.log(
                    {
                        "loss": loss.detach().cpu().item(),
                        "step": step_num,
                        "lr": lr
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
    main(project_name='dqn_drqn_breakout_sandbox', run_name='[GTX970] drqn_sarsd_hparams_fixed')
