import ipdb
import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing


class TimeDependent(MessagePassing):
    def __init__(self):
        super(TimeDependent, self).__init__(aggr='add')
        self.edge_update = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.create_past_msgs = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.create_future_msgs = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.combine_future_past = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, x, edge_index, edge_attr, node_ts, initial_x):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index,
                              size=(x.size(0), x.size(0)),
                              x=x,
                              edge_attr=edge_attr,
                              initial_x=initial_x)

    def message(self, edge_index, x_i, x_j, edge_attr, initial_x_i, initial_x_j):
        """
        edge_index: Tensor of shape (2, E)
        x_i, x_j: Tensor of shape (E, num_node_features)
        edge_attr: Tensor of shape (E, num_edge_features)
        initial_x_i, initial_x_j: Tensor of shape (E, num_node_features)
        """
        # Update the edge features based on the adjacent nodes
        edge_update_features = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edge_attr = self.edge_update(edge_update_features)

        # To construct future messages one takes the feature of the starting
        # node of every edge and combines it with the edge feature
        future_msg_feats = torch.cat([x_i, updated_edge_attr, initial_x_i], dim=1)
        future_msgs = self.create_future_msgs(future_msg_feats)

        # For past messages one takes the feature of the node the edge points to
        past_msg_feats = torch.cat([x_j, updated_edge_attr, initial_x_j], dim=1)
        past_msgs = self.create_past_msgs(past_msg_feats)

        return past_msgs, future_msgs, updated_edge_attr

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        updated_nodes = self.combine_future_past(aggr_out)

        return updated_nodes

    def propagate(self, edge_index, size=None, dim=0, **kwargs):
        """ Code from standard propagate function in PyG """
        dim = 0
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(dim)
                            if size[1 - idx] != tmp[1 - idx].size(dim):
                                raise ValueError('Size Error')
                        tmp = tmp[idx]

                    if tmp is None:
                        message_args.append(tmp)
                    else:
                        if size[idx] is None:
                            size[idx] = tmp.size(dim)
                        if size[idx] != tmp.size(dim):
                            raise ValueError('Size Error')

                        tmp = torch.index_select(tmp, dim, edge_index[idx])
                        message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        """ Message propagation for future and past messages separately """
        past_msgs, future_msgs, update_edges_attr = self.message(*message_args)

        rows, cols = edge_index
        # Edge direction goes in direction of time
        # Send past messages in direction of the edge
        messages_past = scatter_add(past_msgs, cols, dim=0, dim_size=size[1])

        # Send future messages in opposite direction of the edge
        messages_future = scatter_add(future_msgs, rows, dim=0, dim_size=size[0])
        messages = torch.cat([messages_past, messages_future], dim=1)

        return self.update(messages, *update_args), update_edges_attr
