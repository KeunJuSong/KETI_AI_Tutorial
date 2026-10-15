import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class GCN(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=4,
                 num_layers=3, dropout=0.0):
        super().__init__()
        if num_layers < 2 or hidden_channels < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid GCN layer count, width, or dropout.")
        dims = [in_channels] + [hidden_channels] * num_layers
        self.convs = nn.ModuleList(
            GCNConv(left, right) for left, right in zip(dims[:-1], dims[1:])
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.classifier(x)

        return out

class GAT(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=4,
                 num_layers=3, heads=2, dropout=0.0):
        super().__init__()
        if num_layers < 2 or hidden_channels < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid GAT layer count, width, or dropout.")
        dims = [in_channels] + [hidden_channels] * num_layers
        self.convs = nn.ModuleList(
            GATConv(left, right, heads, concat=False) for left, right in zip(dims[:-1], dims[1:])
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.classifier(x)

        return out

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=4,
                 num_layers=3, aggr="mean", dropout=0.0):
        super().__init__()
        if num_layers < 2 or hidden_channels < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid GAT layer count, width, or dropout.")
        dims = [in_channels] + [hidden_channels] * num_layers
        self.convs = nn.ModuleList(
            SAGEConv(left, right, aggr) for left, right in zip(dims[:-1], dims[1:])
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.classifier(x)

        return out
