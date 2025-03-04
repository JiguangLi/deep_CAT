import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class OnlinePredNNV1(nn.Module):
    def __init__(self, pred_stats_dim ,num_items, hidden_dim=1024):
        super(OnlinePredNNV1, self).__init__()
        self.pred_stats_num = pred_stats_dim
        self.input_dim = self.pred_stats_num * num_items
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_items)

    def forward(self, x, mask):
        # x shape: (batch_size, num_items, stats_dim)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 5000)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        output = logits + ((1 - mask) * -1e9)
        return output


class OnlinePredNNV2(nn.Module):
    def __init__(self, pred_stats_dim, num_items, num_heads=8, num_layers=2):
        super(OnlinePredNNV2, self).__init__()
        self.pred_stats_dim = pred_stats_dim
        self.num_items = num_items
        self.input_embedding = nn.Linear(self.pred_stats_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x, mask):
        # x shape: (batch_size, num_itens, stats_dim)
        x = self.input_embedding(x)  # Shape: (batch_size, num_items, 64)
        x = x.permute(1, 0, 2)       # Transformer expects input as (sequence_length, batch_size, embedding_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)       # Back to (batch_size, 500, 64)
        x = self.output_layer(x).squeeze(-1)  # Shape: (batch_size, 500)
        output = x + ((1 - mask) * -1e9)
        return output


class ItemEncoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super(ItemEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x shape: (batch_size, num_items, input_dim)
        return self.encoder(x)  # Output shape: (batch_size, num_items, hidden_dim)


class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim=64):
        super(AttentionMechanism, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, item_embeddings):
        # item_embeddings shape: (batch_size, num_items, hidden_dim)
        Q = self.query(item_embeddings)  # Shape: (batch_size, num_items, hidden_dim)
        K = self.key(item_embeddings)  # Shape: (batch_size, num_items, hidden_dim)
        V = self.value(item_embeddings)  # Shape: (batch_size, num_items, hidden_dim)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (batch_size, num_items, num_items)
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)  # Optional scaling

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (batch_size, num_items, num_items)

        # Compute global context
        global_context = torch.matmul(attention_weights, V)  # Shape: (batch_size, num_items, hidden_dim)

        return global_context


class OnlinePredNNV3(nn.Module):
    def __init__(self, pred_stats_dim, hidden_dim=64, output_dim=1):
        super(OnlinePredNNV3, self).__init__()
        self.pred_stats_dim = pred_stats_dim
        self.item_encoder = ItemEncoder(pred_stats_dim, hidden_dim)
        self.attention = AttentionMechanism(hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, mask):
        # x shape: (batch_size, num_items=500, input_dim=10)
        batch_size, num_items, _ = x.size()

        # Encode items
        item_embeddings = self.item_encoder(x)  # Shape: (batch_size, num_items, hidden_dim)

        # Get global context through attention
        global_context = self.attention(item_embeddings)  # Shape: (batch_size, num_items, hidden_dim)

        # Combine local and global features
        combined_features = torch.cat([item_embeddings, global_context],
                                      dim=-1)  # Shape: (batch_size, num_items, hidden_dim * 2)

        # Compute outputs for each item
        outputs = self.output_layer(combined_features).squeeze(-1)  # Shape: (batch_size, num_items)

        outputs = torch.where(mask.bool(), outputs, torch.tensor(float('-inf')).to(outputs.device))

        #outputs = outputs  + ((1 - mask) * -1e9)
        return outputs
