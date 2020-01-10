import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
from torch_geometric.data import Data
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from src.gnn_tracker.modules.spp import SpatialPyramidPool
from src.gnn_tracker.modules.message_passing import SAGEConv


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

        self.conv1 = SAGEConv(128, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=df.item_id.max() + 1, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data_loader):
        nodes = []
        edges = []
        for t, sample in enumerate(data_loader):
            boxes = sample['gt']
            imgs = sample['cropped_imgs']

            for box_id, img in imgs.items():
                img_encoding = self.appearance_encoder(img.unsqueeze(0))

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x