from random import sample
import logging

class ReplayBuffer:
    def __init__(self, buffer_size=100000, truncate_batch=True, guaranteed_size=3):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.idx = 0
        self.truncate_batch = truncate_batch
        self.guaranteed_size = guaranteed_size

    def insert(self, sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        assert num_samples < min(self.idx, self.buffer_size)
        buf = self.buffer
        if self.idx < self.buffer_size:
            buf = self.buffer[:self.idx]
        random_sample = sample(buf, num_samples)

        # return full mini-batch
        if not self.truncate_batch:
            return random_sample

        # remove elements after first terminal state
        done_list = [s.done for s in random_sample]
        if True in done_list[:self.guaranteed_size-1]:  # check if first guaranteed_size elements don't contain terminal state
            #logging.warning(f'd smaller than {self.guaranteed_size} detected. Re-sampling.')
            return self.sample(num_samples)
        try:
            idx_to_cut = done_list.index(True)
            return random_sample[:idx_to_cut + 1]
        except ValueError:
            return random_sample
