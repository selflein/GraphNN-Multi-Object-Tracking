import torch
from torch import nn

from src.gnn_tracker.modules.message_passing import TimeDependent


class Net(torch.nn.Module):
    def __init__(
            self, 
            num_steps=12, 
            inp_edge_dim=6,
            edge_dim=64, 
            inp_node_dim=512,
            node_dim=32
        ):
        super(Net, self).__init__()
        self.num_steps = num_steps

        self.edge_encoder = nn.Sequential(
            nn.Linear(inp_edge_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, edge_dim)
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.node_encoder = nn.Sequential(
            nn.Linear(inp_node_dim, inp_node_dim // 2),
            nn.ReLU(),
            nn.Linear(inp_node_dim // 2, node_dim)
        )

        self.td1 = TimeDependent(node_dim=node_dim, edge_dim=edge_dim)

    def forward(self, data, initial_x):
        x, edge_index, edge_attr, node_timestamps, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.node_timestamps,
            data.batch
        )

        edge_attr = self.edge_encoder(edge_attr)
        x = self.node_encoder(x)
        initial_x = x.clone()

        for _ in range(self.num_steps):
            x, edge_attr = self.td1(x, edge_index, edge_attr, node_timestamps, initial_x)

        return self.edge_classifier(edge_attr)
