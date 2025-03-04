import torch
import torch.nn as nn


class Phi1Network(nn.Module):
    """Handle Permutation Invariant of the Posterior"""
    def __init__(self, input_dim, hidden_dim):
        super(Phi1Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


class Phi2Network(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Phi2Network, self).__init__()
        self.fc1 = nn.Linear(input_dim,input_dim*2)
        self.fc2 = nn.Linear(input_dim*2,input_dim*2)
        self.fc3 = nn.Linear(input_dim*2, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_items, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.phi1 = Phi1Network(input_dim, hidden_dim)
        self.phi2 = Phi2Network(hidden_dim, num_items)

    def forward(self, x, mask):
        batch_size = x.size(0)
        k = x.size(1) // self.input_dim  # Determine the number of chunks
        combined_phi1 = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for m in range(k):
            x_subset = x[:, (m * self.input_dim):((m + 1) * self.input_dim)]  # Extract each 3-dimensional subset
            # Ignore padding values by setting them to zero before passing through the network
            x_subset = torch.where(x_subset == -1e9, torch.zeros_like(x_subset), x_subset)
            combined_phi1 += self.phi1(x_subset)

        logits = self.phi2(combined_phi1)
        # Apply mask to ensure padding does not affect the output
        masked_logits = logits + ((1-mask) * -1e9)
        return masked_logits


class Phi1StatsNetwork(nn.Module):
    """Handle Permutation Invariant of the Posterior"""
    def __init__(self, input_dim, hidden_dim):
        super(Phi1StatsNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x



class OnlinePolicyNetwork(nn.Module):
    def __init__(self, input_dim, input_stats_dim, num_items, hidden_dim=256):
        super(OnlinePolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.phi1 = Phi1Network(input_dim, hidden_dim)
        self.phi1stats = Phi1StatsNetwork(input_stats_dim, hidden_dim)
        self.phi2 = Phi2Network(hidden_dim*2, num_items)

    def forward(self, x, x_stats, mask):
        batch_size = x.size(0)
        k = x.size(1) // self.input_dim  # Determine the number of chunks
        combined_phi1 = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for m in range(k):
            x_subset = x[:, (m * self.input_dim):((m + 1) * self.input_dim)]  # Extract each 3-dimensional subset
            # Ignore padding values by setting them to zero before passing through the network
            x_subset = torch.where(x_subset == -1e9, torch.zeros_like(x_subset), x_subset)
            combined_phi1 += self.phi1(x_subset)
        phi1_stats_output = self.phi1stats(x_stats)
        phi1_all = torch.cat((combined_phi1, phi1_stats_output), dim=1)
        logits = self.phi2(phi1_all)
        # Apply mask to ensure padding does not affect the output
        masked_logits = logits + ((1-mask) * -1e9)
        return masked_logits

