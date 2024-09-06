import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Embedding(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Embedding, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        #n_input_channels = observation_space.shape[0]

        self.encoder = nn.LSTM(input_size=1, hidden_size=features_dim, num_layers=1)

    def forward(self, observations: th.Tensor) -> th.Tensor:

        '''
        observations.shape = [max_in_seq_len, input_size]
        '''
        lengths = observations[-1, :].tolist()
        lengths = list(map(int, lengths))
        observations = observations[:-1, :]

        observations = th.unsqueeze(observations, dim=0)  # [batch_size, max_in_seq_len, input_size]

        observations = th.transpose(observations, 1, 0)   # [max_in_seq_len, batch_size, input_size]

        packed_src = pack(observations, lengths, enforce_sorted=False)

        encoder_output, hidden_final = self.encoder(packed_src, None)

        out = th.squeeze(hidden_final[0], dim=1)

        return out



class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper


        self.embedding = nn.Embedding(vocab_size=2000, embedding_dim=features_dim)
        self.encoder = nn.LSTM(input_size=features_dim, hidden_size=features_dim, num_layers=1)


    def forward(self, observations: th.Tensor) -> th.Tensor:

        output, (hidden, cell) = self.encoder(self.embedding(observations))
        print(output.shape)
        print(hidden.shape)
        print(cell.shape)
        return output

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )
# model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
# model.learn(1000)

# net = Embedding()
# input_data=th.rand(size=(64, 281))
# output = net(input_data)
# print(output)
