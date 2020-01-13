import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

import ipdb
class TimeDependent(MessagePassing):
    def __init__(self):
        super(TimeDependent, self).__init__(aggr='max')

    def forward(self, x, edge_index, edge_attributes, node_timestamps):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        ipdb.set_trace(context=10)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, e=edge_attributes, node_timestamps=node_timestamps)

    def message(self, x_i, x_j, e, node_timestamps):
        # x_j has shape [E, in_channels]

        ipdb.set_trace(context=10)
        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        ipdb.set_trace(context=10)
        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding
