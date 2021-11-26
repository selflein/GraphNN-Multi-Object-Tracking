import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing


class TimeDependent(MessagePassing):
    """
    Implements time-aware message passing as proposed in

    """

    def __init__(self, node_dim=32, edge_dim=64):
        super(TimeDependent, self).__init__(aggr="add", node_dim=0)

        # Update the edge embedding using
        # * adajacent node features
        # * current edge features
        # according to Equation 3.
        edge_input_dim = 2 * node_dim + edge_dim
        self.edge_update = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim // 2),
            nn.ReLU(),
            nn.Linear(edge_input_dim // 2, edge_dim),
        )

        # Create message using
        # * initial node features (at the start of message) passing
        # * current node features
        # * features of the edge for which the message is created
        # Depending on whether the connected node is in the past or future
        # `create_past_msgs` or `create_future_msgs` networks are used.
        # See Equation 6.
        node_input_dim = 2 * node_dim + edge_dim
        self.create_past_msgs = nn.Sequential(
            nn.Linear(node_input_dim, node_input_dim // 2),
            nn.ReLU(),
            nn.Linear(node_input_dim // 2, node_dim),
        )
        self.create_future_msgs = nn.Sequential(
            nn.Linear(node_input_dim, node_input_dim // 2),
            nn.ReLU(),
            nn.Linear(node_input_dim // 2, node_dim),
        )

        # Combines the seperately aggregated future and past messages
        # according to Equation 9 to get the updated node embedding.
        self.combine_future_past = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr, node_ts, initial_x):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(
            edge_index,
            size=(x.size(0), x.size(0)),
            x=x,
            edge_attr=edge_attr,
            initial_x=initial_x,
        )

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

    def update(self, inputs):
        messages, edge_attr = inputs
        # Create new node embeddings based on Equation 9.
        updated_nodes = self.combine_future_past(messages)
        return updated_nodes, edge_attr

    def aggregate(self, inputs, edge_index, dim_size):
        past_msgs, future_msgs, edge_attr = inputs

        rows, cols = edge_index
        # Edge direction goes in direction of time
        # Send past messages in direction of the edge
        messages_past = scatter(
            past_msgs, cols, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr
        )

        # Send future messages in opposite direction of the edge
        messages_future = scatter(
            future_msgs, rows, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr
        )

        # Concatenate messages (see Equation 9)
        messages = torch.cat([messages_past, messages_future], dim=1)
        return messages, edge_attr
