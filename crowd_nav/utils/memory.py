from torch.utils.data import Dataset
import numpy as np
import math
import torch
from tqdm import tqdm
import networkx as nx
import bz2
import _pickle as cPickle
import os


class ReplayMemory(Dataset):
    def __init__(self, capacity, window_size, device):
        self.capacity = capacity
        self.window_size = window_size
        self.device = device
        self.memory = list()
        self.position = 0
        self.data_path = "/home/cbl/HSR-V1/memory_buffer"
        #self.path_list = os.listdir(self.data_path)
    
    def push(self, item):
        # replace old experience with new experience
        #if len(self.memory) < self.position + 1:
        #    self.memory.append(item)
        #else:
        #    self.memory[self.position] = item
        robot_states, human_states, local_map, actions, values, rewards, next_robot_states, next_human_states, next_local_map\
                                                                    ,mask, recurrent_hidden_state = item
        path = f'{self.data_path}/{self.position:04d}.pbz2'
        with bz2.BZ2File(path, 'w') as fp:
            cPickle.dump(
                {
                    'robot_states': robot_states,
                    'human_states': human_states,
                    'local_map': local_map,
                    'actions': actions,
                    'values': values,
                    'rewards': rewards,
                    'next_robot_states': next_robot_states,
                    'next_human_states': next_human_states,
                    'next_local_map': next_local_map,
                    'mask': mask,
                    'recurrent_hidden_state': recurrent_hidden_state
                },
                fp
            )
        if len(self.memory) < self.position + 1:
            self.memory.append(self.position)
        else:
            self.memory[self.position] = self.position
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    
    def __getitem__(self, item):
        #name = self.path_list[item]
        #item_path = os.path.join(self.data_path, name)
        robot_states_ = torch.Tensor().to(self.device)
        human_states_ = torch.Tensor().to(self.device)
        local_map_ = torch.Tensor().to(self.device)
        actions_ = torch.Tensor().to(self.device)
        values_ = torch.Tensor().to(self.device)
        rewards_ = torch.Tensor().to(self.device)
        next_robot_states_ = torch.Tensor().to(self.device)
        next_human_states_ = torch.Tensor().to(self.device)
        next_local_map_ = torch.Tensor().to(self.device)
        mask_ = torch.Tensor().to(self.device)
        recurrent_hidden_state_ = {
            'local_map_rnn_v': torch.Tensor().to(self.device),
            'agent_node_rnn_v': torch.Tensor().to(self.device),
            'robot_node_rnn_v': torch.Tensor().to(self.device),
            'final_node_rnn_v': torch.Tensor().to(self.device),
            'local_map_rnn_s': torch.Tensor().to(self.device),
            'agent_node_rnn_s': torch.Tensor().to(self.device),
            'robot_node_rnn_s': torch.Tensor().to(self.device),
        }
        for i in range(item, item + self.window_size):
            item_path = f'{self.data_path}/{i:04d}.pbz2'
            with bz2.BZ2File(item_path, 'rb') as fp:
                data = cPickle.load(fp)
                robot_states = data['robot_states']
                human_states = data['human_states']
                local_map = data['local_map']
                actions = data['actions']
                values = data['values']
                rewards = data['rewards']
                next_robot_states = data['next_robot_states']
                next_human_states = data['next_human_states']
                next_local_map = data['next_local_map']
                mask = data['mask']
                recurrent_hidden_state = data['recurrent_hidden_state']
            robot_states_ = torch.cat([robot_states_, robot_states.unsqueeze(0)], dim=0)
            human_states_ = torch.cat([human_states_, human_states.unsqueeze(0)], dim=0)
            local_map_ = torch.cat([local_map_, local_map.unsqueeze(0).to(self.device)], dim=0)
            actions_ = torch.cat([actions_, actions.unsqueeze(0).to(self.device)], dim=0)
            values_ = torch.cat([values_, values.unsqueeze(0).to(self.device)], dim=0)
            rewards_ = torch.cat([rewards_, rewards.unsqueeze(0).to(self.device)], dim=0)
            next_robot_states_ = torch.cat([next_robot_states_, next_robot_states.unsqueeze(0)], dim=0)
            next_human_states_ = torch.cat([next_human_states_, next_human_states.unsqueeze(0)], dim=0)
            next_local_map_ = torch.cat([next_local_map_, next_local_map.unsqueeze(0).to(self.device)], dim=0)
            mask_ = torch.cat([mask_, mask.unsqueeze(0).to(self.device)], dim=0)

            for k, v in recurrent_hidden_state_.items():
                recurrent_hidden_state_[k] = torch.cat([v, recurrent_hidden_state[k].to(self.device)], dim=0)

        return (robot_states_, human_states_, local_map_, actions_, values_, rewards_, next_robot_states_,
                                next_human_states_, next_local_map_, mask_, recurrent_hidden_state_)
        #return self.memory[item]

    
    def __len__(self):
        return len(self.memory) - self.window_size

    def clear(self):
        self.memory = list()
