import torch
from torch.utils.data import Dataset
import numpy as np

class OddballDataset(Dataset):
    """
    Synthetic dataset for Oddball paradigm.
    Generates sequences of tokens where one token is frequent (standard) 
    and another is rare (oddball).
    """
    def __init__(self, num_samples, seq_len, p_oddball=0.2, 
                 standard_token=0, oddball_token=1, 
                 context_switch=False, seed=None):
        """
        Args:
            num_samples (int): Number of sequences to generate.
            seq_len (int): Length of each sequence.
            p_oddball (float): Probability of the oddball token.
            standard_token (int): Token ID for standard stimulus.
            oddball_token (int): Token ID for oddball stimulus.
            context_switch (bool): If True, flips probabilities halfway through.
            seed (int): Random seed.
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.p_oddball = p_oddball
        self.standard_token = standard_token
        self.oddball_token = oddball_token
        self.context_switch = context_switch
        
        if seed is not None:
            np.random.seed(seed)
            
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            seq = []
            for t in range(self.seq_len):
                # Determine current probability
                current_p = self.p_oddball
                
                if self.context_switch and t >= self.seq_len // 2:
                    # Switch context: Standard becomes rare, Oddball becomes frequent
                    # Or just invert the probability? Let's invert p_oddball.
                    # If p was 0.1 (rare), now it becomes 0.9 (frequent).
                    current_p = 1.0 - self.p_oddball
                
                if np.random.rand() < current_p:
                    token = self.oddball_token
                else:
                    token = self.standard_token
                seq.append(token)
            data.append(np.array(seq))
        return torch.LongTensor(np.array(data))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input: x_0 ... x_{T-1}
        # Target: x_1 ... x_T
        seq = self.data[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y
