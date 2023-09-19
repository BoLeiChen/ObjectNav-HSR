import os
import logging
import copy
import torch
from tqdm import tqdm
from HRO_sim.envs.utils.info import *
import numpy as np

class Explorer(object):
    def __init__(self, env, robot, device, writer, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        collisionobs_times = []
        timeout_times = []
        success = 0
        collision = 0
        collisionobs = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        SPL = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        collisionobs_cases = []
        timeout_cases = []

        if k != 1:
            pbar = tqdm(total=k)
        else:
            pbar = None

        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            rpx, rpy = self.robot.get_position()  # robot px, py
            path_length = 0.0
            states = []
            actions = []
            rewards = []
            local_maps = []
            masks = [torch.FloatTensor([1.0])]
            recurrent_hidden_states = []
            robot_num = 1
            agent_num = 20
            patch_size = 8 ** 2
            rhs = {}
            rhs['local_map_rnn_v'] = torch.zeros(1, 1, patch_size, 64)
            rhs['agent_node_rnn_v'] = torch.zeros(1, 1, agent_num, 64)
            rhs['robot_node_rnn_v'] = torch.zeros(1, 1, robot_num, 64)
            rhs['final_node_rnn_v'] = torch.zeros(1, 1, robot_num, 256)
            rhs['local_map_rnn_s'] = torch.zeros(1, 1, patch_size, 64)
            rhs['agent_node_rnn_s'] = torch.zeros(1, 1, agent_num, 64)
            rhs['robot_node_rnn_s'] = torch.zeros(1, 1, robot_num, 64)
            recurrent_hidden_states.append(rhs)

            while not done:
                action, rnn_hxs = self.robot.act(ob, masks[-1], recurrent_hidden_states[-1], imitation_learning)
                ob, reward, done, info = self.env.step(action)

                rpx_1, rpy_1 = self.robot.get_position()
                path_length = path_length + np.linalg.norm(np.array([rpx - rpx_1, rpy - rpy_1])) * 0.05
                rpx, rpy = rpx_1, rpy_1

                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                local_maps.append(ob['local_map'])
                masks.append(torch.FloatTensor([0.0]) if done else torch.FloatTensor([1.0]))
                recurrent_hidden_states.append(rnn_hxs)

                if isinstance(info, Discomfort):
                    discomfort += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, Discomfort):
                discomfort += 1
                min_dist.append(info.min_dist)
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
                SPL.append(path_length / max(self.env.length, path_length))
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, CollisionObs):
                collisionobs += 1
                collisionobs_cases.append(i)
                collisionobs_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            video_file = None
            self.env.render('traj', video_file)

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision) or isinstance(info, CollisionObs):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, local_maps, masks, recurrent_hidden_states, imitation_learning)
            #print(len(self.memory))

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / k
        collision_rate = collision / k
        collisionobs_rate = collisionobs / k
        spl = sum(SPL) / k
        assert success + collision + timeout + collisionobs == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, collisionobs rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f},'
                     'average return: {:.4f}, SPL: {:.4f}'. format(phase.upper(), extra_info, success_rate, collision_rate, collisionobs_rate,
                                                       avg_nav_time, average(cumulative_rewards),
                                                       average(average_returns), spl))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times)
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         discomfort / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        self.statistics = success_rate, collision_rate, collisionobs_rate, avg_nav_time, average(cumulative_rewards), average(average_returns)
        return self.statistics

    def update_memory(self, states, actions, rewards, local_maps, masks, recurrent_hidden_states, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            local_map = local_maps[i]
            next_local_map = local_maps[i+1]
            mask = masks[i+1]
            recurrent_hidden_state = recurrent_hidden_states[i+1]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i+1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value])
            reward = torch.Tensor([rewards[i]])
            action = torch.Tensor([actions[i]])

            if self.target_policy.name == 'ModelPredictiveRL':
                self.memory.push((state[0], state[1], local_map, action, value, reward, next_state[0], next_state[1],
                                  next_local_map, mask, recurrent_hidden_state))
            else:
                self.memory.push((state, value, reward, next_state))

    def log(self, tag_prefix, global_step):
        sr, cr, cor, time, reward, avg_return = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/collisionobs_rate', cor, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
