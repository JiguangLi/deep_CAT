import random
from collections import deque
import numpy as np
import pickle


class ReplayBuffer:
    """Queue storing transitions (s,a, r,s')"""
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        """Experience: (cur_pos, action, reward, next_pos, done)

        cur_pos/next_pos: (pos_paramter, running_statistics, boolean_mask)

        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        raw_samples = random.sample(self.buffer, batch_size)
        max_len_cur_pos = max([raw_samples[i][0][0].shape[0] for i in range(batch_size)])
        max_len_next_pos = max([raw_samples[i][3][0].shape[0] for i in range(batch_size)])
        padded_samples = [(
            (np.pad(m[0][0], (0, max_len_cur_pos - len(m[0][0])), mode='constant'), m[0][1], m[0][2]),
            m[1],
            m[2],
            (np.pad(m[3][0], (0, max_len_next_pos - len(m[3][0])), mode='constant'), m[3][1], m[3][2]),
            m[4]
        ) for m in raw_samples]
        return padded_samples

    def size(self):
        return len(self.buffer)


class OnlineReplayBuffer:
    """Queue storing transitions (s,a, r,s'), including online predictions"""
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        """Experience: (cur_pos, action, reward, next_pos, done)

        cur_pos/next_pos: (pos_paramter, running_statistics, predictions, boolean_mask)
                            preds_only case: (predictions, boolean_mask)
        """
        self.buffer.append(experience)

    def sample(self, batch_size, pred_only = False):
        raw_samples = random.sample(self.buffer, batch_size)
        if not pred_only:
            max_len_cur_pos = max([raw_samples[i][0][0].shape[0] for i in range(batch_size)])
            max_len_next_pos = max([raw_samples[i][3][0].shape[0] for i in range(batch_size)])
            padded_samples = [(
                (np.pad(m[0][0], (0, max_len_cur_pos - len(m[0][0])), mode='constant'), m[0][1], m[0][2], m[0][3]),
                m[1],
                m[2],
                (np.pad(m[3][0], (0, max_len_next_pos - len(m[3][0])), mode='constant'), m[3][1], m[3][2], m[3][3]),
                m[4]
            ) for m in raw_samples]
        else:
            padded_samples = [((m[0][0], m[0][1]), m[1], m[2], (m[3][0], m[3][1]), m[4]) for m in raw_samples]
        return padded_samples

    def size(self):
        return len(self.buffer)

    def save(self, filepath):
        """Save the replay buffer to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filepath):
        """Load the replay buffer from a file."""
        with open(filepath, 'rb') as f:
            self.buffer = pickle.load(f)


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer storing transitions (s,a,r,s') with priorities."""
    def __init__(self, buffer_size, alpha):
        self.buffer_size = buffer_size
        self.alpha = alpha  # Controls the level of prioritization
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0  # Initial maximum priority

    def add(self, experience):
        """Add a new experience with maximum priority."""
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            # Replace the oldest experience if buffer is full
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)

    def sample(self, batch_size, beta, pred_only=False):
        if len(self.buffer) == 0:
            return [], [], [], []

        # Calculate sampling probabilities
        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        raw_samples = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        # Process samples (padding if necessary)
        if not pred_only:
            max_len_cur_pos = max([raw_samples[i][0][0].shape[0] for i in range(batch_size)])
            max_len_next_pos = max([raw_samples[i][3][0].shape[0] for i in range(batch_size)])
            padded_samples = [(
                (np.pad(m[0][0], (0, max_len_cur_pos - len(m[0][0])), mode='constant'), m[0][1], m[0][2], m[0][3]),
                m[1],
                m[2],
                (np.pad(m[3][0], (0, max_len_next_pos - len(m[3][0])), mode='constant'), m[3][1], m[3][2], m[3][3]),
                m[4]
            ) for m in raw_samples]
        else:
            padded_samples = [((m[0][0], m[0][1]), m[1], m[2], (m[3][0], m[3][1]), m[4]) for m in raw_samples]

        return padded_samples, indices, weights, probs[indices]

    def update_priorities(self, indices, priorities):
        """Update the priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def size(self):
        return len(self.buffer)

    def save(self, filepath):
        """Save the replay buffer and priorities to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.buffer, self.priorities), f)

    def load(self, filepath):
        """Load the replay buffer and priorities from a file."""
        with open(filepath, 'rb') as f:
            self.buffer, self.priorities = pickle.load(f)
