import torch
import torch.nn as nn
import numpy as np


class Phi1Network(nn.Module):
    """Handle Permutation Invariant of the Posterior"""
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.3):
        super(Phi1Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        if x.size(0) ==  1:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(torch.relu(self.fc3(x)))
        else:
            x = torch.relu(self.bn1(self.fc1(x)))
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        return x


class Phi1StatsNetwork(nn.Module):
    """Input running statistics"""
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.3):
        super(Phi1StatsNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        return x


class Phi1PredsNetwork(nn.Module):
    """Input predictions"""
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.3):
        super(Phi1PredsNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # New additional hidden layer
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))  # Additional hidden layer
        return x


class SmallPhi2Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_prob=0.3):
        super(SmallPhi2Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(self.fc3(x))  # No activation in the output layer
        return x


class NewOnlineQNetwork(nn.Module):
    """Adding OnlinePrediction Mean and Variance"""
    def __init__(self, input_dim, input_stats_dim, num_items, hidden_dim_phi1=256, dropout_prob=0.3):
        super(NewOnlineQNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_phi1 = hidden_dim_phi1
        self.phi1 = Phi1Network(input_dim, hidden_dim_phi1, dropout_prob)
        self.hidden_dim_stats = input_stats_dim
        self.phi1stats = Phi1StatsNetwork(input_stats_dim, self.hidden_dim_stats, 0.1)
        self.hidden_dim_preds = num_items
        self.phi1preds = Phi1PredsNetwork(2*num_items, self.hidden_dim_preds, dropout_prob)
        self.hidden_phi2_dim = 2**int(np.log2(num_items)+1)
        self.phi2 = SmallPhi2Network(hidden_dim_phi1 + self.hidden_dim_stats + self.hidden_dim_preds,
                                     self.hidden_phi2_dim,
                                     num_items, dropout_prob)

    def forward(self, x, x_stats, x_preds, mask):
        batch_size = x.size(0)
        k = x.size(1) // self.input_dim  # Determine the number of chunks
        combined_phi1 = torch.zeros(batch_size, self.hidden_dim_phi1).to(x.device)
        for m in range(k):
            x_subset = x[:, (m * self.input_dim):((m + 1) * self.input_dim)]
            is_zero_input = (x_subset == 0).all(dim=1)
            output = torch.zeros(x_subset.size(0), self.phi1.fc3.out_features, device=x_subset.device)
            if not is_zero_input.all():
                non_zero_output = self.phi1(x_subset[~is_zero_input])
                output[~is_zero_input, :] = non_zero_output
            combined_phi1 = combined_phi1 + output
        phi1_stats_output = self.phi1stats(x_stats)
        phi1_preds_output = self.phi1preds(x_preds)
        phi1_all = torch.cat((combined_phi1, phi1_stats_output, phi1_preds_output), dim=1)
        logits = self.phi2(phi1_all)
        masked_logits = logits + ((1-mask) * -1e9)
        return masked_logits

