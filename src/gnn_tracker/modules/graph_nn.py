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
    def __init__(self):
        super(Net, self).__init__()
        self.appearance_encoder = AppearanceEncoder([8, 4, 1])

        self.edge_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64)
        )

        self.td1 = TimeDependent()

    def forward(self, data):
        x, edge_index, edge_attr, node_timestamps, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.node_timestamps,
            data.batch
        )

        edge_attr = self.edge_encoder(edge_attr)

        x = self.td1(x, edge_index, edge_attr, node_timestamps)
        return x
