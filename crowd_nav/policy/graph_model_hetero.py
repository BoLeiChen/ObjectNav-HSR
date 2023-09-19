from crowd_nav.utils.pyHGT.pyHGT.model import GNN as PyHGT_GNN
from HRO_sim.envs.HRO_sim import classes_PONI
from typing import *
import numpy as np
import logging
import torch
from torch.nn import functional as F
import math
import time

import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from HRO_sim.envs.utils.utils import visualize_map
import matplotlib.pyplot as plt

SEM_CLASSES = classes_PONI
GLOVE_DIM = 50

def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,\
            padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,\
            bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MapConv_down(nn.Module):

    def __init__(self, PyHGTModel_S=False, num_channel=23, conf=False, rc=False, layers=[2, 2, 2, 2]):
        super(MapConv_down, self).__init__()

        self.conf = conf
        self.rc = rc
        self.num_channel = num_channel

        block = BasicBlock
        self.inplanes = 64

        self.conv1 = nn.Conv2d(num_channel + int(self.conf) + int(self.rc), self.inplanes, kernel_size=7, stride=2, padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        if PyHGTModel_S:
            self.conv3 = nn.Conv2d(128, 23, kernel_size=3, stride=2, padding=1)
        else:
            self.conv3 = nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        x = self.conv2(x)
        x = self.relu(x)
        #x = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        x = self.conv3(x)

        return x

class MapConv_up(nn.Module):

    def __init__(self, num_channel = 23):
        super(MapConv_up, self).__init__()

        self.conv1 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(64, 23, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):    #(32, 8, 8)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.relu(x)     #(64, 16, 16)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)     #(128, 32, 32)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv3(x)
        x = self.relu(x)     #(64, 64, 64)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)    #(23, 128, 128)

        return x


class SelfAttentionBlock(nn.Module):

    def __init__(self, key_in_channels, query_in_channels, out_channels, channels=None,
                 share_key_query=True, matmul_norm=True, with_out=False):
        super(SelfAttentionBlock, self).__init__()
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.matmul_norm = matmul_norm
        self.channels = channels
        self.key_project = nn.Linear(key_in_channels, channels if with_out else out_channels)
        self.value_project = nn.Linear(key_in_channels, channels if with_out else out_channels)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = nn.Linear(query_in_channels, channels if with_out else out_channels)
        self.out_proj = None
        if with_out:
            self.out_proj = nn.Linear(channels, out_channels)
        else:
            self.channels = out_channels

    def forward(self, query_feats, key_feats=None):
        if key_feats is None:
            key_feats = query_feats
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)

        sim_map = torch.matmul(key, query)
        if self.matmul_norm:
            sim_map = (self.channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).sum(dim=-1).contiguous()

        if self.out_proj is not None:
            context = self.out_proj(context)

        return context

class RNNBase(nn.Module):
    # edge: True -> edge RNN, False -> node RNN
    def __init__(self, input_size = 32, output_size = 256):
        super(RNNBase, self).__init__()
        self.gru = nn.GRU(input_size, output_size)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            # use env dimension as batch
            # [1, 12, 6, ?] -> [1, 12*6, ?] or [30, 6, 6, ?] -> [30, 6*6, ?]
            seq_len, nenv, agent_num, _ = x.size()
            x = x.view(seq_len, nenv*agent_num, -1)
            hxs_times_masks = hxs * (masks.view(seq_len, nenv, 1, 1))
            hxs_times_masks = hxs_times_masks.view(seq_len, nenv*agent_num, -1)
            x, hxs = self.gru(x, hxs_times_masks) # we already unsqueezed the inputs in SRNN forward function
            x = x.view(seq_len, nenv, agent_num, -1)
            hxs = hxs.view(seq_len, nenv, agent_num, -1)

        # during update, input shape[0] * nsteps (30) = hidden state shape[0]
        else:

            # N: nenv, T: seq_len, agent_num: node num or edge num
            T, N, agent_num, _ = x.size()
            # x = x.view(T, N, agent_num, x.size(2))

            # Same deal with masks
            masks = masks.view(T, N)

            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            # hxs = hxs.unsqueeze(0)
            # hxs = hxs.view(hxs.size(0), hxs.size(1)*hxs.size(2), hxs.size(3))
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                # x and hxs have 4 dimensions, merge the 2nd and 3rd dimension
                x_in = x[start_idx:end_idx]
                x_in = x_in.view(x_in.size(0), x_in.size(1)*x_in.size(2), x_in.size(3))
                hxs = hxs.view(hxs.size(0), N, agent_num, -1)
                hxs = hxs * (masks[start_idx].view(1, -1, 1, 1))
                hxs = hxs.view(hxs.size(0), hxs.size(1) * hxs.size(2), hxs.size(3))
                rnn_scores, hxs = self.gru(x_in, hxs)

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T, N, agent_num, -1)
            hxs = hxs.view(1, N, agent_num, -1)
        return x, hxs

class HeteroBase:
    def compute_obstacle_embedding(self, sem_class_idx, approx):
        sem_class_one_hot = torch.zeros(self.num_sem_class, dtype=torch.float).to(self.device)
        sem_class_one_hot[sem_class_idx + 1] = 1

        if self.use_sem_mixin:
            arr = torch.FloatTensor([]).to(self.device)
            for p in approx:
                arr = torch.cat([arr, p[0].to(self.device).unsqueeze(0)])
                #arr = torch.cat((arr, torch.cat((sem_class_one_hot, p[0].to(self.device))).unsqueeze(0)))
            return self.obstacle_linear(self.obstacle_attn(arr.unsqueeze(0)))[0]
        else:
            arr = []
            for p in approx:
                arr.append(torch.Tensor(p[0]).to(self.device))
            context = self.obstacle_attn(torch.Tensor([arr]).to(self.device))
            return self.obstacle_linear(torch.cat(sem_class_one_hot, context))[0]

    def compute_obstalce_embedding_vec(self, sem_class_idx, approx_list):
        sem_class_one_hot = torch.zeros((sem_class_idx.shape[0], self.num_sem_class), dtype=torch.float).to(self.device)
        sem_class_one_hot[:, sem_class_idx + 1] = 1

        if self.use_sem_mixin:
            arr = torch.stack([
                [torch.cat((sem_class_one_hot, p[0])) for p in approx] for approx in approx_list
            ]).to(self.device)
        else:
            ...

    def get_obstacle_feature(self, obstacle_one_hot):
        if not self.use_glove:
            return obstacle_one_hot
        else:
            raise NotImplementedError

    def get_target_one_hot(self, local_map_or_target):
        is_local_map = len(local_map_or_target.shape) == 3  # torch.Size([46, 128, 128])
        if is_local_map:
            target_map = local_map_or_target[..., self.num_sem_class:, :, :]  
            sum_target_map = torch.sum(target_map, dim=(-1, -2))  
            target_one_hot = torch.clip(sum_target_map, min=0.0, max=1.0) 
            return target_one_hot.clone().contiguous()
        else:
            return local_map_or_target.clone().contiguous()

    def compute_robot_embedding(self, robot_state, local_map, h, mask):
        if not self.use_target_as_feature:
            robot_feature = self.relu(self.robot_linear1(robot_state.unsqueeze(0))).unsqueeze(0).unsqueeze(0)
            robot_feature, h = self.gru1(robot_feature, h.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
            return self.relu(self.robot_linear2(robot_feature)), h
        else:
            target_one_hot = self.get_target_one_hot(local_map)
            robot_feature = self.relu(self.robot_linear1(torch.cat((robot_state, target_one_hot), dim=-1)).unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            robot_feature, h = self.gru1(robot_feature, h.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
            return self.relu(self.robot_linear2(robot_feature)), h

    def compute_human_embedding(self, human_state, h, mask):
        human_feature = self.relu(self.human_linear1(human_state)).unsqueeze(0).unsqueeze(0)
        human_feature, h = self.gru2(human_feature, h.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
        return self.relu(self.human_linear2(human_feature)), h

    def compute_map_embedding(self, map_state, h, mask):
        map_feature, h = self.gru4(map_state.unsqueeze(0).unsqueeze(0), h.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
        return self.relu(self.map_linear(map_feature)), h

    def compute_map_embedding_s(self, map_state, h, mask):
        map_state = self.relu(self.map_linear1(map_state)).unsqueeze(0).unsqueeze(0)
        map_feature, h = self.gru4(map_state, h.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
        return self.relu(self.map_linear2(map_feature)), h

class PyHGTModel_V(nn.Module, HeteroBase):

    def __init__(self, config=None, in_dim=32, output_size=256, n_heads=8, n_layers=2, obstacle_attn_dim=64,
                 human_state_dim=5, robot_state_dim=9, conv_name='hgt', use_RTE=False, use_time=False, use_glove=False,
                 use_target_as_feature=False, use_sem_mixin=True, sem_classes=SEM_CLASSES, device='cuda', **kw):
        super(PyHGTModel_V, self).__init__()
        self.gnn = PyHGT_GNN(
            in_dim=in_dim,  
            n_hid=in_dim,
            num_types=len(self._node_type_IDs),
            num_relations=len(self._edge_type_IDs),
            n_heads=n_heads,
            n_layers=n_layers,
            conv_name=conv_name,
            use_RTE=use_RTE,
            adapt_in_dim=False,
        )
        self.config = config
        self.time_step = config.hgt.batch_size
        self.use_sem_mixin = use_sem_mixin  
        self.num_sem_class = len(sem_classes) 
        self.sem_classes = sem_classes  
        self._wall_sem_idx = sem_classes.index("wall") - 1
        self.use_target_as_feature = use_target_as_feature  

        self.device = device
        self.use_RTE = use_RTE 
        self.use_time = use_time
        self.use_glove = use_glove
        self.sem_feature_dim = self.num_sem_class if not use_glove else GLOVE_DIM 

        self.gru1 = RNNBase(in_dim, config.HRORNN.node_rnn_size)
        self.gru2 = RNNBase(in_dim, config.HRORNN.node_rnn_size)
        self.gru4 = RNNBase(in_dim, config.HRORNN.node_rnn_size)
        self.gru3 = RNNBase(config.HRORNN.final_rnn_size * 2, config.HRORNN.final_rnn_size)

        self.human_linear1 = nn.Linear(human_state_dim, in_dim)
        self.human_linear2 = nn.Linear(config.HRORNN.node_rnn_size, in_dim)

        if use_target_as_feature:
            self.robot_linear1 = nn.Linear(robot_state_dim + self.sem_feature_dim, in_dim)
        else:
            self.robot_linear1 = nn.Linear(robot_state_dim, in_dim)
        self.robot_linear2 = nn.Linear(config.HRORNN.node_rnn_size, in_dim)

        if use_sem_mixin:
            #self.obstacle_attn = SelfAttentionBlock(self.sem_feature_dim + 2, self.sem_feature_dim + 2,
            #                                        obstacle_attn_dim)
            self.obstacle_attn = SelfAttentionBlock(2, 2, obstacle_attn_dim)
            self.obstacle_linear = nn.Linear(obstacle_attn_dim, in_dim)
        else:
            self.obstacle_attn = SelfAttentionBlock(2, 2, obstacle_attn_dim)
            self.obstacle_linear = nn.Linear(obstacle_attn_dim + self.sem_feature_dim, in_dim)

        #self.predict_attn = SelfAttentionBlock(in_dim, in_dim, output_size)
        self.robot_embedding = nn.Linear(robot_state_dim, output_size)
        self.robot_embedding1 = nn.Linear(in_dim, output_size)
        self.relu = nn.ReLU()

        self.map_linear = nn.Linear(config.HRORNN.node_rnn_size, in_dim)
        self.mapconv = MapConv_down(num_channel=46, conf=False, rc=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, 32, 128 // 16, 128 // 16))

    def forward(self, state, masks, rnn_hxs, local_map_or_target):

        '''
        pos_emd = torch.zeros([state[0].shape[0], 2, 128 // 16, 128 // 16])
        for k in range(state[0].shape[0]):
            for i in range(-(128 // 16 // 2), (128 // 16 // 2)):
                for j in range(-(128 // 16 // 2), (128 // 16 // 2)):
                    rx, ry = state[0][k][0][0], state[0][k][0][1]
                    pos_emd[k, 1, i + (128 // 16 // 2), j + (128 // 16 // 2)] = ry + (2 * i + 1) * (128 // 16) * 0.05
                    pos_emd[k, 0, i + (128 // 16 // 2), j + (128 // 16 // 2)] = rx + (2 * j + 1) * (128 // 16) * 0.05
        '''
        '''
        figure, axes = plt.subplots()
        global_map_vis = visualize_map(local_map_or_target[0][1:23, ...].cpu())
        plt.imshow(global_map_vis)
        # plt.imshow(self.semmap[i + 1, ...])
        pos_emd_0 = pos_emd[0, ...] / 0.05
        for i in range(-4, 4):
            for j in range(-4, 4):
                plt.plot(pos_emd[0, 0, i + 4, j + 4].cpu() / 0.05, pos_emd[0, 1, i + 4, j + 4].cpu() / 0.05, 'o')
                plt.plot([])
        plt.plot(state[0][0][0][0].cpu() / 0.05, state[0][0][0][1].cpu() / 0.05, '*')
        draw_circle = plt.Circle((state[0][0][0][0].cpu() / 0.05, state[0][0][0][1].cpu() / 0.05), 5 / 0.05, fill=False)
        axes.set_aspect(1)
        axes.add_artist(draw_circle)
        plt.show()
        '''

        local_map_feature = self.mapconv(local_map_or_target) + self.pos_embed
        #local_map_feature = torch.cat([local_map_feature, pos_emd.to(self.device)], dim=1)

        batch_size = len(state[0])
        node_feature_list = []
        node_type_list = []
        edge_time_list = []
        edge_index_list = []
        edge_type_list = []
        node_num_of_batch = [] 
        accumulate_node_num = 0 
        h_r_featrues = torch.Tensor().to(self.device)
        h_h_featrues = torch.Tensor().to(self.device)
        h_m_featrues = torch.Tensor().to(self.device)
        for batch_idx in range(batch_size):
            s, lm, lmf = (state[0][batch_idx], state[1][batch_idx]), local_map_or_target[batch_idx], local_map_feature[batch_idx]
            if state[0].shape[0] == rnn_hxs['robot_node_rnn_v'].shape[0]:
                node_feature, node_type, edge_time, edge_index, edge_type, h_r_featrue, h_h_feature, h_m_feature = \
                        self.build_GNN_input(lmf, s, rnn_hxs['agent_node_rnn_v'][batch_idx, ...], rnn_hxs['robot_node_rnn_v'][batch_idx, ...],
                                             rnn_hxs['local_map_rnn_v'][batch_idx, ...], masks[batch_idx, ...], lm)
                h_r_featrues = torch.cat([h_r_featrues, h_r_featrue], dim=1)
                h_h_featrues = torch.cat([h_h_featrues, h_h_feature], dim=1)
                h_m_featrues = torch.cat([h_m_featrues, h_m_feature], dim=1)
            else:
                if batch_idx % self.time_step == 0:
                    h_r_featrue = rnn_hxs['robot_node_rnn_v'][batch_idx // self.time_step, ...]
                    h_h_feature = rnn_hxs['agent_node_rnn_v'][batch_idx // self.time_step, ...]
                    h_m_feature = rnn_hxs['local_map_rnn_v'][batch_idx // self.time_step, ...]
                node_feature, node_type, edge_time, edge_index, edge_type, h_r_featrue, h_h_feature, h_m_feature = \
                        self.build_GNN_input(lmf, s, h_h_feature, h_r_featrue, h_m_feature, masks[batch_idx, ...], lm)
                if (batch_idx + 1) % self.time_step == 0:
                    h_r_featrues = torch.cat([h_r_featrues, h_r_featrue], dim=1)
                    h_h_featrues = torch.cat([h_h_featrues, h_h_feature], dim=1)
                    h_m_featrues = torch.cat([h_m_featrues, h_m_feature], dim=1)
                h_r_featrue = h_r_featrue[0,0,...]
                h_h_feature = h_h_feature[0,0,...]
                h_m_feature = h_m_feature[0, 0, ...]
            for eid in range(len(edge_index)):
                edge_index[eid] = [edge_index[eid][0]+accumulate_node_num, edge_index[eid][1]+accumulate_node_num]
            accumulate_node_num += len(node_feature)
            node_num_of_batch.append(len(node_feature))

            node_feature = torch.stack(node_feature).to(self.device)  # node_num * in_dim
            node_type = torch.LongTensor(node_type).to(self.device)  # node_num * 1
            edge_time = torch.LongTensor(edge_time).to(self.device)  # edge_num * 1
            edge_index = torch.LongTensor(edge_index).to(self.device).t()  # 2 * edge_num
            edge_type = torch.LongTensor(edge_type).to(self.device)  # edge_num * 1
            node_feature_list.append(node_feature)
            node_type_list.append(node_type)
            edge_time_list.append(edge_time)
            edge_index_list.append(edge_index)
            edge_type_list.append(edge_type)

        node_feature = torch.cat(node_feature_list)
        node_type = torch.cat(node_type_list)
        edge_time = torch.cat(edge_time_list)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_type = torch.cat(edge_type_list)
        node_rep = self.gnn(node_feature, node_type, edge_time, edge_index, edge_type)
        node_reps = torch.split(node_rep, node_num_of_batch, dim=0)             

        values = torch.Tensor().to(self.device)
        for i in range(len(node_reps)):
            values = torch.cat([values, node_reps[i][0,:].unsqueeze(0)], dim=0)
        values = self.relu(self.robot_embedding1(values))

        robot_emb = self.relu(self.robot_embedding(state[0]))
        if state[0].shape[0] == rnn_hxs['robot_node_rnn_v'].shape[0]:
            # (1, N, 1, 32)   (1, N, 1, 256)
            values = torch.cat([values.unsqueeze(0).unsqueeze(2), robot_emb.unsqueeze(0)], dim= -1)
            values, h_f_featrues = self.gru3(values, rnn_hxs['final_node_rnn_v'].unsqueeze(0), masks.unsqueeze(0))
            values = values.squeeze(2).squeeze(0)
        else:
            # (30, nenv, 1, 32)  (1, nenv, 1, 256)
            seq_length = self.time_step
            nenv = state[0].shape[0] // self.time_step
            values = self.reshapeT(values.unsqueeze(1), seq_length, nenv)
            masks = self.reshapeT(masks, seq_length, nenv)
            robot_emb = self.reshapeT(robot_emb, seq_length, nenv)
            values = torch.cat([values, robot_emb], dim=-1)
            values, h_f_featrues = self.gru3(values, rnn_hxs['final_node_rnn_v'].unsqueeze(0), masks)
            shape = values.size()[2:]
            values = values.reshape(seq_length * nenv, *shape).squeeze(1)

        rnn_hxs_ = {}
        rnn_hxs_['agent_node_rnn_v'] = h_h_featrues.squeeze(0)
        rnn_hxs_['robot_node_rnn_v'] = h_r_featrues.squeeze(0)
        rnn_hxs_['final_node_rnn_v'] = h_f_featrues.squeeze(0)
        rnn_hxs_['local_map_rnn_v'] = h_m_featrues.squeeze(0)

        return values, rnn_hxs_

    _node_type_IDs = {
        "robot": 0, "human": 1, "object": 2, "wall": 3
    }
    _edge_type_IDs = {
        "self": 0, "w2r": 1, "h2r": 2, "o2r": 3, "w2h": 4, "o2h": 5, "h2h": 6, "o2o": 7, "r2r": 8
    }  # r:robot, h:human, o:object, w:wall, 2:to

    def reshapeT(self, T, seq_length, nenv):
        shape = T.size()[1:]
        return T.unsqueeze(0).reshape((seq_length, nenv, *shape))

    def type_id(self, type_name) -> int:
        
        if type_name in self._node_type_IDs:
            return self._node_type_IDs[type_name]
        elif type_name in self._edge_type_IDs:
            return self._edge_type_IDs[type_name]

    def build_GNN_input(self, local_map_feature, state, h_agent, h_robot, h_map, mask, local_map_or_target):

        robot_state, human_state = state
        node_feature = []  
        node_type = [] 
        node_time = [] 
        edge_index = [] 
        edge_type = [] 
        edge_time = []  
        # robot_state is of shape [batch_size,  1, 9]
        # human_state is of shape [batch_size, num_human, 3]
        # observable_obstacles is list of [sem_class, approx, num_points, 0, 2]

        start_index_of_robot_node = 0
        for r in robot_state:
            f, h_r_feature = self.compute_robot_embedding(r, local_map_or_target, h_robot, mask)
            node_feature.append(f[0,0,0,...])
            node_type.append(self.type_id("robot"))

        start_index_of_human_node = len(node_feature)
        f, h_h_feature = self.compute_human_embedding(human_state, h_agent, mask)
        f = f[0,0,...]
        for i in range(f.shape[0]):
            node_feature.append(f[i, ...])
            node_type.append(self.type_id("human"))

        start_index_of_object_node = len(node_feature)
        L, H, W = local_map_feature.size()
        local_map_feature = local_map_feature.reshape(L, H * W).t()
        f, h_m_feature = self.compute_map_embedding(local_map_feature, h_map, mask)
        f = f[0,0,...]
        for i in range(f.shape[0]):
            node_feature.append(f[i, ...])
            node_type.append(self.type_id("object"))
        robot_nodes = range(start_index_of_robot_node, start_index_of_human_node)
        human_nodes = range(start_index_of_human_node, start_index_of_object_node)
        object_nodes = range(start_index_of_object_node, len(node_feature))

        # robot to robot
        for i, robot_node_1 in enumerate(robot_nodes):
            for j, robot_node_2 in enumerate(robot_nodes):
                edge_index.append([robot_node_1, robot_node_2])
                edge_type.append(self.type_id("r2r"))

        # human to robot & human
        for i, human_node in enumerate(human_nodes):
            for j, robot_node in enumerate(robot_nodes):
                edge_index.append([human_node, robot_node])
                edge_type.append(self.type_id("h2r"))
            for j, human_node_2 in enumerate(human_nodes):
                edge_index.append([human_node, human_node_2])
                edge_type.append(self.type_id("h2h"))

        # object to human&robot&object
        for object_node in object_nodes:
            for human_node in human_nodes:
                edge_index.append([object_node, human_node])
                edge_type.append(self.type_id("o2h"))
            for robot_node in robot_nodes:
                edge_index.append([object_node, robot_node])
                edge_type.append(self.type_id("o2r"))
            for object_node_2 in object_nodes:
                edge_index.append([object_node, object_node_2])
                edge_type.append(self.type_id("o2o"))


        return node_feature, node_type, edge_time, edge_index, edge_type, h_r_feature, h_h_feature, h_m_feature

class PyHGTModel_S(nn.Module, HeteroBase):

    def __init__(self, config=None, in_dim=32, output_size=256, n_heads=8, n_layers=2, obstacle_attn_dim=64,
                 human_state_dim=5, robot_state_dim=9, conv_name='hgt', use_RTE=False, use_time=False, use_glove=False,
                 use_target_as_feature=False, use_sem_mixin=True, sem_classes=SEM_CLASSES, device='cuda', **kw):
        super(PyHGTModel_S, self).__init__()
        self.gnn = PyHGT_GNN(
            in_dim=in_dim, 
            n_hid=in_dim,
            num_types=len(self._node_type_IDs),
            num_relations=len(self._edge_type_IDs),
            n_heads=n_heads,
            n_layers=n_layers,
            conv_name=conv_name,
            use_RTE=use_RTE,
            adapt_in_dim=False,
        )
        self.config = config
        self.time_step = config.hgt.batch_size
        self.use_sem_mixin = use_sem_mixin 
        self.num_sem_class = len(sem_classes) 
        self.sem_classes = sem_classes
        self._wall_sem_idx = sem_classes.index("wall") - 1  
        self.use_target_as_feature = use_target_as_feature  

        self.device = device
        self.use_RTE = use_RTE  
        self.use_time = use_time  
        self.use_glove = use_glove
        self.sem_feature_dim = self.num_sem_class if not use_glove else GLOVE_DIM  

        self.gru1 = RNNBase(in_dim, config.HRORNN.node_rnn_size)
        self.gru2 = RNNBase(in_dim, config.HRORNN.node_rnn_size)
        self.gru4 = RNNBase(in_dim, config.HRORNN.node_rnn_size)
        self.gru3 = RNNBase(config.HRORNN.final_rnn_size * 2, config.HRORNN.final_rnn_size)

        self.human_linear1 = nn.Linear(human_state_dim, in_dim)
        self.human_linear2 = nn.Linear(config.HRORNN.node_rnn_size, in_dim)

        if use_target_as_feature:
            self.robot_linear1 = nn.Linear(robot_state_dim + self.sem_feature_dim, in_dim)
        else:
            self.robot_linear1 = nn.Linear(robot_state_dim, in_dim)
        self.robot_linear2 = nn.Linear(config.HRORNN.node_rnn_size, in_dim)

        if use_sem_mixin:
            #self.obstacle_attn = SelfAttentionBlock(self.sem_feature_dim + 2, self.sem_feature_dim + 2,
            #                                        obstacle_attn_dim)
            self.obstacle_attn = SelfAttentionBlock(2, 2, obstacle_attn_dim)
            self.obstacle_linear = nn.Linear(obstacle_attn_dim, in_dim)
        else:
            self.obstacle_attn = SelfAttentionBlock(2, 2, obstacle_attn_dim)
            self.obstacle_linear = nn.Linear(obstacle_attn_dim + self.sem_feature_dim, in_dim)

        #self.predict_attn = SelfAttentionBlock(in_dim, in_dim, output_size)
        self.robot_embedding = nn.Linear(robot_state_dim, output_size)
        self.robot_embedding1 = nn.Linear(in_dim, output_size)
        self.relu = nn.ReLU()

        self.map_linear1 = nn.Linear(23, in_dim)
        self.map_linear2 = nn.Linear(config.HRORNN.node_rnn_size, in_dim)
        self.mapconv = MapConv_down(True, num_channel=23, conf=False, rc=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, 23, 128 // 16, 128 // 16))

    def forward(self, state, masks, rnn_hxs, local_map_or_target):

        local_map_feature = self.mapconv(local_map_or_target[:, 0:23, ...]) + self.pos_embed
        #local_map_feature = torch.cat([local_map_feature, pos_emd.to(self.device)], dim=1)

        batch_size = len(state[0])
        node_feature_list = []
        node_type_list = []
        edge_time_list = []
        edge_index_list = []
        edge_type_list = []
        node_num_of_batch = [] 
        accumulate_node_num = 0
        h_r_featrues = torch.Tensor().to(self.device)
        h_h_featrues = torch.Tensor().to(self.device)
        h_m_featrues = torch.Tensor().to(self.device)
        for batch_idx in range(batch_size):
            s, lm, lmf = (state[0][batch_idx], state[1][batch_idx]), local_map_or_target[batch_idx], local_map_feature[batch_idx]
            if state[0].shape[0] == rnn_hxs['robot_node_rnn_s'].shape[0]:
                node_feature, node_type, edge_time, edge_index, edge_type, h_r_featrue, h_h_feature, h_m_feature = \
                        self.build_GNN_input(lmf, s, rnn_hxs['agent_node_rnn_s'][batch_idx, ...], rnn_hxs['robot_node_rnn_s'][batch_idx, ...],
                                             rnn_hxs['local_map_rnn_s'][batch_idx, ...], masks[batch_idx, ...], lm)
                h_r_featrues = torch.cat([h_r_featrues, h_r_featrue], dim=1)
                h_h_featrues = torch.cat([h_h_featrues, h_h_feature], dim=1)
                h_m_featrues = torch.cat([h_m_featrues, h_m_feature], dim=1)
            else:
                if batch_idx % self.time_step == 0:
                    h_r_featrue = rnn_hxs['robot_node_rnn_s'][batch_idx // self.time_step, ...]
                    h_h_feature = rnn_hxs['agent_node_rnn_s'][batch_idx // self.time_step, ...]
                    h_m_feature = rnn_hxs['local_map_rnn_s'][batch_idx // self.time_step, ...]
                node_feature, node_type, edge_time, edge_index, edge_type, h_r_featrue, h_h_feature, h_m_feature = \
                        self.build_GNN_input(lmf, s, h_h_feature, h_r_featrue, h_m_feature, masks[batch_idx, ...], lm)
                if (batch_idx + 1) % self.time_step == 0:
                    h_r_featrues = torch.cat([h_r_featrues, h_r_featrue], dim=1)
                    h_h_featrues = torch.cat([h_h_featrues, h_h_feature], dim=1)
                    h_m_featrues = torch.cat([h_m_featrues, h_m_feature], dim=1)
                h_r_featrue = h_r_featrue[0,0,...]
                h_h_feature = h_h_feature[0,0,...]
                h_m_feature = h_m_feature[0, 0, ...]
            for eid in range(len(edge_index)):
                edge_index[eid] = [edge_index[eid][0]+accumulate_node_num, edge_index[eid][1]+accumulate_node_num]
            accumulate_node_num += len(node_feature)
            node_num_of_batch.append(len(node_feature))

            node_feature = torch.stack(node_feature).to(self.device)  # node_num * in_dim
            node_type = torch.LongTensor(node_type).to(self.device)  # node_num * 1
            edge_time = torch.LongTensor(edge_time).to(self.device)  # edge_num * 1
            edge_index = torch.LongTensor(edge_index).to(self.device).t()  # 2 * edge_num
            edge_type = torch.LongTensor(edge_type).to(self.device)  # edge_num * 1
            node_feature_list.append(node_feature)
            node_type_list.append(node_type)
            edge_time_list.append(edge_time)
            edge_index_list.append(edge_index)
            edge_type_list.append(edge_type)

        node_feature = torch.cat(node_feature_list)
        node_type = torch.cat(node_type_list)
        edge_time = torch.cat(edge_time_list)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_type = torch.cat(edge_type_list)
        node_rep = self.gnn(node_feature, node_type, edge_time, edge_index, edge_type)
        node_reps = list(torch.split(node_rep, node_num_of_batch, dim=0))            

        rnn_hxs_ = {}
        rnn_hxs_['agent_node_rnn_s'] = h_h_featrues.squeeze(0)
        rnn_hxs_['robot_node_rnn_s'] = h_r_featrues.squeeze(0)
        rnn_hxs_['local_map_rnn_s'] = h_m_featrues.squeeze(0)

        return node_reps, rnn_hxs_

    _node_type_IDs = {
        "robot": 0, "human": 1, "object": 2, "wall": 3
    }
    _edge_type_IDs = {
        "self": 0, "w2r": 1, "h2r": 2, "o2r": 3, "w2h": 4, "o2h": 5, "h2h": 6, "o2o": 7, "r2r": 8
    }  # r:robot, h:human, o:object, w:wall, 2:to

    def reshapeT(self, T, seq_length, nenv):
        shape = T.size()[1:]
        return T.unsqueeze(0).reshape((seq_length, nenv, *shape))

    def type_id(self, type_name) -> int:
        if type_name in self._node_type_IDs:
            return self._node_type_IDs[type_name]
        elif type_name in self._edge_type_IDs:
            return self._edge_type_IDs[type_name]

    def build_GNN_input(self, local_map_feature, state, h_agent, h_robot, h_map, mask, local_map_or_target):

        robot_state, human_state = state
        node_feature = [] 
        node_type = [] 
        node_time = [] 
        edge_index = [] 
        edge_type = []  
        edge_time = [] 
        # robot_state is of shape [batch_size,  1, 9]
        # human_state is of shape [batch_size, num_human, 3]
        # observable_obstacles is list of [sem_class, approx, num_points, 0, 2]

        start_index_of_robot_node = 0
        for r in robot_state:
            f, h_r_feature = self.compute_robot_embedding(r, local_map_or_target, h_robot, mask)
            node_feature.append(f[0,0,0,...])
            node_type.append(self.type_id("robot"))

        start_index_of_human_node = len(node_feature)
        f, h_h_feature = self.compute_human_embedding(human_state, h_agent, mask)
        f = f[0,0,...]
        for i in range(f.shape[0]):
            node_feature.append(f[i, ...])
            node_type.append(self.type_id("human"))

        start_index_of_object_node = len(node_feature)
        L, H, W = local_map_feature.size()
        local_map_feature = local_map_feature.reshape(L, H * W).t()
        f, h_m_feature = self.compute_map_embedding_s(local_map_feature, h_map, mask)
        f = f[0,0,...]
        for i in range(f.shape[0]):
            node_feature.append(f[i, ...])
            node_type.append(self.type_id("object"))
        robot_nodes = range(start_index_of_robot_node, start_index_of_human_node)
        human_nodes = range(start_index_of_human_node, start_index_of_object_node)
        object_nodes = range(start_index_of_object_node, len(node_feature))

   
        for i, robot_node_1 in enumerate(robot_nodes):
            for j, robot_node_2 in enumerate(robot_nodes):
                edge_index.append([robot_node_1, robot_node_2])
                edge_type.append(self.type_id("r2r"))

        for i, human_node in enumerate(human_nodes):
            for j, robot_node in enumerate(robot_nodes):
                edge_index.append([human_node, robot_node])
                edge_type.append(self.type_id("h2r"))
            for j, human_node_2 in enumerate(human_nodes):
                edge_index.append([human_node, human_node_2])
                edge_type.append(self.type_id("h2h"))

    
        for object_node in object_nodes:
            for human_node in human_nodes:
                edge_index.append([object_node, human_node])
                edge_type.append(self.type_id("o2h"))
            for robot_node in robot_nodes:
                edge_index.append([object_node, robot_node])
                edge_type.append(self.type_id("o2r"))
            for object_node_2 in object_nodes:
                edge_index.append([object_node, object_node_2])
                edge_type.append(self.type_id("o2o"))

        return node_feature, node_type, edge_time, edge_index, edge_type, h_r_feature, h_h_feature, h_m_feature