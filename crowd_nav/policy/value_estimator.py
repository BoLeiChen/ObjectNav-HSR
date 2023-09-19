import torch.nn as nn
from crowd_nav.policy.helpers import mlp
import torch

class ValueEstimator(nn.Module):
    def __init__(self, config, graph_model):
        super().__init__()
        self.graph_model = graph_model
        self.value_network = mlp(config.hgt.X_dim_v, config.hgt.value_network_dims)

    def forward(self, state, masks, rnn_hxs):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3
        assert len(state[2].shape) == 4

        # only use the feature of robot node as state representation
        state_embedding, rnn_hxs = self.graph_model([state[0], state[1]], masks, rnn_hxs, state[2])
        value = self.value_network(state_embedding)
        return value, rnn_hxs
