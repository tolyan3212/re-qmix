import numpy as np


class ReplayBuffer:
    def __init__(
            self,
            max_steps,
            sequence_steps,
            sequences_overlapping
    ):
        self.max_steps = max_steps
        self.sequence_steps = sequence_steps
        self.sequences_overlapping = sequences_overlapping
        if self.sequences_overlapping:
            self.step = self.sequence_steps // 2
        else:
            self.step = self.sequence_steps

        self.max_buffer_size = self.max_steps // self.sequence_steps
        self.array = np.array([None] * self.max_buffer_size, dtype=object)

        self.current_buffer_size = 0
        self.current_buffer_index = 0

    def add(self, episode):
        for i in range(0, len(episode), self.step):
            if self.sequences_overlapping:
                # Case when algorithm works with burn-in and the
                # length of the remaining sequence is less then half
                # of the sequence size will result in storing a
                # sequence, at which training will only happen at the
                # empty part of it
                if len(episode) - i <= self.sequence_steps // 2:
                    break
            sequence = episode[i:min(i+self.sequence_steps, len(episode))]
            self.__add_sequence(sequence)

    def __add_sequence(self, sequence):
        self.array[self.current_buffer_index] = sequence
        self.current_buffer_index += 1
        self.current_buffer_index %= self.max_buffer_size
        if self.current_buffer_size < self.max_buffer_size:
            self.current_buffer_size += 1

    def can_sample(self, batch_size):
        return batch_size <= self.current_buffer_size

    def sample(self, batch_size):
        assert batch_size <= self.current_buffer_size

        seq_ids = np.random.choice(np.arange(self.current_buffer_size),
                                   batch_size)
        return self.array[seq_ids]


class EpisodesBuffer:
    def __init__(
            self,
            episodes_count,
    ):
        self.max_episodes_count = episodes_count

        self.array = np.array([None] * self.max_episodes_count, dtype=object)

        self.current_buffer_size = 0
        self.current_buffer_index = 0

    def add(self, episode):
        self.array[self.current_buffer_index] = episode
        self.current_buffer_index += 1
        self.current_buffer_index %= self.max_episodes_count
        if self.current_buffer_size < self.max_episodes_count:
            self.current_buffer_size += 1

    def can_sample(self, batch_size):
        return batch_size <= self.current_buffer_size

    def sample(self, batch_size):
        if batch_size > self.current_buffer_size:
            raise RuntimeError(f'EpisodeBuffer.sample(): given batch size ({batch_size}) is larger than current buffer size ({self.current_buffer_size})')

        seq_ids = np.random.choice(np.arange(self.current_buffer_size),
                                   batch_size)
        return self.array[seq_ids]


class ContinuousBuffer:
    def __init__(
            self,
            max_steps,
            sequence_steps,
    ):
        self.max_steps = max_steps
        self.sequence_steps = sequence_steps
        self.array = np.array([None] * self.max_steps, dtype=object)

        self.current_buffer_size = 0
        self.current_buffer_index = 0

    def add(self, episode):
        for i, step in enumerate(episode):
            self.array[self.current_buffer_index] = step
            self.current_buffer_index += 1
            self.current_buffer_index %= self.max_steps
            if self.current_buffer_size < self.max_steps:
                self.current_buffer_size += 1

    def can_sample(self, batch_size):
        return batch_size * self.sequence_steps <= self.current_buffer_size

    def sample(self, batch_size):
        if batch_size * self.sequence_steps > self.current_buffer_size:
            raise RuntimeError(f'ContinuousBuffer.sample(): given batch size ({batch_size} x {self.sequence_steps}) is larger than current buffer size ({self.current_buffer_size})')

        batch = np.array([None] * batch_size, dtype=object)

        seq_starts = np.random.choice(np.arange(self.current_buffer_size),
                                      batch_size)
        for i, start in enumerate(seq_starts):
            if start + self.sequence_steps < self.current_buffer_size:
                batch[i] = self.array[start:start+self.sequence_steps]
            else:
                seq = list(self.array[start:self.current_buffer_size])
                rest_elements = self.sequence_steps - (self.current_buffer_size - start)
                seq.extend(list(self.array[:rest_elements]))
                batch[i] = np.array(seq)

        return batch
