import torch
from torch import nn
from torchvision.models import vgg16_bn

from src.gnn_tracker.modules.spp import SpatialPyramidPool
from src.gnn_tracker.modules.message_passing import TimeDependent


class AppearanceEncoder(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.feats = vgg16_bn(pretrained=True).features[:14]
        print(self.feats)
        self.pool = SpatialPyramidPool(levels)

    def forward(self, inp):
        b, c, h, w = inp.shape
        out = self.feats(inp)
        out = self.pool(out)
        return out


class Net(torch.nn.Module):
    def __init__(self, num_steps=12, edge_features=64):
        super(Net, self).__init__()
        self.num_steps = num_steps
        # self.appearance_encoder = AppearanceEncoder([8, 4, 1])

        self.edge_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, edge_features)
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.td1 = TimeDependent()

    def forward(self, data, initial_x):
        x, edge_index, edge_attr, node_timestamps, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.node_timestamps,
            data.batch
        )

        edge_attr = self.edge_encoder(edge_attr)

        for _ in range(self.num_steps):
            x, edge_attr = self.td1(x, edge_index, edge_attr, node_timestamps, initial_x)

        return self.edge_classifier(edge_attr)
