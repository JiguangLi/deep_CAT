import torch
import torch.nn as nn
import numpy as np


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


class Phi1StatsNetwork(nn.Module):
    """Input running statistics"""
    def __init__(self, input_dim, hidden_dim):
        super(Phi1StatsNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class QNetwork(nn.Module):
    def __init__(self, input_dim, input_stats_dim, num_items, hidden_dim_phi1=256, hidden_dim_stats=256):
        super(QNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_phi1 = hidden_dim_phi1
        self.hidden_dim_stats = hidden_dim_stats
        self.phi1 = Phi1Network(input_dim, hidden_dim_phi1)
        self.phi1stats = Phi1StatsNetwork(input_stats_dim, hidden_dim_stats)
        self.phi2 = Phi2Network(hidden_dim_phi1+hidden_dim_stats, num_items)

    def forward(self, x, x_stats, mask):
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
        phi1_all = torch.cat((combined_phi1, phi1_stats_output), dim=1)
        logits = self.phi2(phi1_all)
        masked_logits = logits + ((1-mask) * -1e9)
        return masked_logits


class Phi1PredsNetwork(nn.Module):
    """Input running statistics"""
    def __init__(self, input_dim, hidden_dim):
        super(Phi1PredsNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class SmallPhi2Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SmallPhi2Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class OnlineQNetwork(nn.Module):
    """Adding OnlinePrediction Mean and Variance"""
    def __init__(self, input_dim, input_stats_dim, num_items, hidden_dim_phi1=256):
        super(OnlineQNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_phi1 = hidden_dim_phi1
        self.phi1 = Phi1Network(input_dim, hidden_dim_phi1)
        #self.hidden_dim_stats = 2**int(np.log2(input_stats_dim)+1)
        self.hidden_dim_stats = input_stats_dim
        self.phi1stats = Phi1StatsNetwork(input_stats_dim, self.hidden_dim_stats)
        #self.hidden_dim_preds = 2**int(np.log2(num_items*2)+1)
        self.hidden_dim_preds = num_items
        self.phi1preds = Phi1PredsNetwork(2*num_items, self.hidden_dim_preds)
        self.hidden_phi2_dim = 2**int(np.log2(num_items)+1)
        self.phi2 = SmallPhi2Network(hidden_dim_phi1 + self.hidden_dim_stats + self.hidden_dim_preds,
                                     self.hidden_phi2_dim,
                                     num_items)

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


class Phi1PredsNetworkV2(nn.Module):
    """Don't apply activation in the end, one more hidden layers"""
    def __init__(self, input_dim, hidden_dim):
        super(Phi1PredsNetworkV2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OnlineQNetworkV2(nn.Module):
    """Adding OnlinePrediction Mean and Variance: assuming additivity of state and prediction network in the final

     This Performs very badly for some reason - didn't converge
     """
    def __init__(self, input_dim, input_stats_dim, num_items, hidden_dim_phi1=256):
        super(OnlineQNetworkV2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_phi1 = hidden_dim_phi1
        self.phi1 = Phi1Network(input_dim, hidden_dim_phi1)
        self.hidden_dim_stats = input_stats_dim
        self.phi1stats = Phi1StatsNetwork(input_stats_dim, self.hidden_dim_stats)
        self.hidden_phi2_dim = 2 * num_items
        self.phi2 = SmallPhi2Network(hidden_dim_phi1 + self.hidden_dim_stats, self.hidden_phi2_dim,
                                     num_items)
        self.phi1predsvar = Phi1PredsNetworkV2(num_items, num_items)
        self.phi1predsmean = Phi1PredsNetworkV2(num_items, num_items)

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
        state_latent = torch.cat((combined_phi1, phi1_stats_output), dim=1)
        state_value = self.phi2(state_latent)
        num_items = int(x_preds.shape[1]/2)
        pred1_value = self.phi1predsvar(x_preds[:, :num_items])
        pred2_value = self.phi1predsmean(x_preds[:, -num_items:])
        total_value = state_value + pred1_value + pred2_value
        masked_logits = total_value + ((1-mask) * -1e9)
        return masked_logits


class OnlineQNetworkV3(nn.Module):
    """Adding OnlinePrediction Mean and Variance: same as original one, but adding predictions in the end

    Don't assume simple linear additivity assumation: use predictions twice!!!
    """
    def __init__(self, input_dim, input_stats_dim, num_items, hidden_dim_phi1=256):
        super(OnlineQNetworkV3, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_phi1 = hidden_dim_phi1
        self.phi1 = Phi1Network(input_dim, hidden_dim_phi1)
        self.hidden_dim_stats = input_stats_dim
        self.phi1stats = Phi1StatsNetwork(input_stats_dim, self.hidden_dim_stats)
        self.hidden_dim_preds = num_items
        self.phi1preds = Phi1PredsNetwork(2 * num_items, self.hidden_dim_preds)
        self.hidden_phi2_dim = 2 * num_items
        self.phi2 = SmallPhi2Network(hidden_dim_phi1 + self.hidden_dim_stats + self.hidden_dim_preds,
                                     self.hidden_phi2_dim,
                                     num_items)
        self.direct_weights1 = nn.Parameter(torch.randn(1,num_items)) # for variance
        self.direct_weights2 = nn.Parameter(torch.randn(1,num_items)) # for mean

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
        shared_output = self.phi2(phi1_all)
        num_items = int(x_preds.shape[1]/2)
        pred1_value = self.direct_weights1 * x_preds[:, :num_items]
        pred2_value = self.direct_weights2 * x_preds[:, -num_items:]
        total_value = shared_output + pred1_value + pred2_value
        masked_logits = total_value + ((1-mask) * -1e9)
        return masked_logits


class RowwiseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_items):
        super(RowwiseNetwork, self).__init__()
        # Define a network to process each row independently
        self.row_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single value per row
        )

        self.num_items = num_items

    def forward(self, X):
        # X has shape [batch_size, num_items, 11]
        # Process each row independently
        row_outputs = [self.row_network(X[:, i, :]) for i in range(self.num_items)]  # Apply row_network to each row
        # Combine row outputs into a tensor of shape [batch_size, num_items]
        combined_output = torch.cat(row_outputs, dim=1)
        return combined_output


class OnlineQNetworkV4(nn.Module):
    """Similar to V2, but include prediction quantiles rather than the variances and the means.

    This Performs very badly for some reason - didn't converge.
    """
    def __init__(self, input_dim, input_stats_dim, num_items, hidden_dim_phi1=256):
        super(OnlineQNetworkV4, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_phi1 = hidden_dim_phi1
        self.phi1 = Phi1Network(input_dim, hidden_dim_phi1)
        self.hidden_dim_stats = input_stats_dim
        self.phi1stats = Phi1StatsNetwork(input_stats_dim, self.hidden_dim_stats)
        self.hidden_phi2_dim = 2 * num_items
        self.phi2 = SmallPhi2Network(hidden_dim_phi1 + self.hidden_dim_stats, self.hidden_phi2_dim,
                                     num_items)
        self.pred_network = RowwiseNetwork(input_dim=11, hidden_dim=256, num_items=num_items)
        # New network to combine state_value and pred_value
        self.combiner_network = nn.Sequential(
            nn.Linear(2 * num_items, num_items),
            nn.ReLU(),
            nn.Linear(num_items, num_items)  # Output a single value per item
        )

    def forward(self, x, x_stats, x_preds, mask):
        """X_preds have dimension (BATCH_size, num_items, 11)"""
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
        state_latent = torch.cat((combined_phi1, phi1_stats_output), dim=1)
        state_value = self.phi2(state_latent)
        pred_value = self.pred_network(x_preds)
        combined_input = torch.cat((state_value, pred_value), dim=1)  # Concatenate along the feature dimension
        total_value = self.combiner_network(combined_input)
        mask = mask.to(dtype=torch.bool)
        masked_logits = torch.where(mask == 1, total_value, torch.tensor(float('-inf'), device=total_value.device))
        return masked_logits


class OnlineQNetworkV5(nn.Module):
    """Modified V4 to include nonlinear processing of x_preds."""

    def __init__(self, input_dim, input_stats_dim, num_items, hidden_dim_phi1=256):
        super(OnlineQNetworkV5, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_phi1 = hidden_dim_phi1
        self.num_items = num_items
        self.phi1 = Phi1Network(input_dim, hidden_dim_phi1)

        self.hidden_dim_stats = input_stats_dim
        self.phi1stats = Phi1StatsNetwork(input_stats_dim, self.hidden_dim_stats)

        # Add nonlinear processing of x_preds
        self.hidden_dim_preds = hidden_dim_phi1
        self.phi1preds = Phi1PredsNetwork(11, self.hidden_dim_preds)  # Processing the 11-dimensional x_preds

        # Final network combining all features
        self.hidden_phi2_dim = hidden_dim_phi1 + self.hidden_dim_stats + self.hidden_dim_preds
        self.phi2 = SmallPhi2Network(self.hidden_phi2_dim, hidden_dim_phi1, num_items)

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

        # Process x_stats
        phi1_stats_output = self.phi1stats(x_stats)

        # Process x_preds nonlinearly
        x_preds_reshaped = x_preds.view(batch_size * self.num_items, 11)  # Process each x_preds row separately
        phi1_preds_output = self.phi1preds(x_preds_reshaped)
        phi1_preds_output = phi1_preds_output.view(batch_size, self.num_items, -1).mean(dim=1)  # Aggregate over num_items

        # Combine all features
        state_latent = torch.cat((combined_phi1, phi1_stats_output, phi1_preds_output), dim=1)
        state_value = self.phi2(state_latent)

        # Mask outputs
        masked_logits = state_value + ((1 - mask) * -1e9)
        return masked_logits
