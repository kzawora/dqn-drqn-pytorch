import cv2
import numpy as np


class BreakoutEnv:
    def __init__(self, env, w, h):
        self.env = env
        self.w = w
        self.h = h
        self.buffer = np.zeros((1, h, w)).astype(np.uint8)
        self.frame = None

    def _preprocess_frame(self, frame):
        img = cv2.resize(frame, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def step(self, action):
        img, reward, done, info = self.env.step(action)
        self.frame = img.copy()
        img = self._preprocess_frame(img)
        self.buffer[0, :, :] = img
        return self.buffer.copy(), reward, done, info

    @property
    def observation_space(self):
        return np.zeros((1, self.h, self.w))

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        img = self.env.reset()
        self.frame = img.copy()
        img = self._preprocess_frame(img)
        self.buffer = np.stack([img], 0)
        return self.buffer.copy()

    def render(self, mode='human'):
        self.env.render(mode)


class BreakoutFrameStackingEnv:
    def __init__(self, env, w, h, num_stack=4):
        self.env = env
        self.n = num_stack
        self.w = w
        self.h = h

        self.buffer = np.zeros((num_stack, h, w)).astype(np.uint8)
        self.frame = None

    def _preprocess_frame(self, frame):
        img = cv2.resize(frame, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def step(self, action):
        img, reward, done, info = self.env.step(action)
        self.frame = img.copy()
        img = self._preprocess_frame(img)
        self.buffer[1:self.n, :, :] = self.buffer[0:self.n - 1, :, :]
        self.buffer[0, :, :] = img
        return self.buffer.copy(), reward, done, info

    @property
    def observation_space(self):
        return np.zeros((self.n, self.h, self.w))

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        img = self.env.reset()
        self.frame = img.copy()
        img = self._preprocess_frame(img)
        self.buffer = np.stack([img] * self.n, 0)
        return self.buffer.copy()

    def render(self, mode='human'):
        self.env.render(mode)
