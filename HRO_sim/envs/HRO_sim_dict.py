import gym
import numpy as np
from numpy.linalg import norm
import copy
import random
import torch
from HRO_sim.envs.utils.action import ActionRot, ActionXY
from HRO_sim.envs import HROSim
from HRO_sim.envs.utils.utils import point_to_segment_dist, visualize_map, plot_points, plot_path
import logging
import matplotlib.pyplot as plt
from HRO_sim.envs.utils.info import *
import time
from HRO_sim.envs.utils.state import ObservableState, ObservableState_nov, FullState
from HRO_sim.envs.policy.policy_factory import policy_factory

classes_PONI = ["floor", "wall", "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed", "chest_of_drawers",
                "plant", "sink", "toilet", "stool", "towel", "tv_monitor", "shower", "bathtub", "counter", "fireplace",
                "gym_equipment", "seating", "clothes"]

class HROSimDict(HROSim):
    def __int__(self):
        super().__init__()

    # update the robot belief of human states
    # if a human is visible, its state is updated to its current ground truth state
    # else we assume it keeps going in a straight line with last observed velocity
    def update_last_human_states(self, human_visibility):
        """
        update the self.last_human_states array
        human_visibility: list of booleans returned by get_human_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        # keep the order of 5 humans at each timestep
        for i in range(self.human_num):
            if human_visibility[i]:
                humanS = np.array(self.humans[i].get_observable_state_list())
                self.last_human_states[i, :] = humanS
            else:
                px, py, vx, vy, r = self.last_human_states[i, :]
                # Plan A: linear approximation of human's next position
                #px = px + vx * self.time_step
                #py = py + vy * self.time_step
                #self.last_human_states[i, :] = np.array([px, py, vx, vy, r])

                #Plan B: assume the human doesn't move, use last observation
                self.last_human_states[i, :] = np.array([px, py, 0., 0., r])

    def get_num_human_in_local_map(self):
        human_ids = []
        humans_in_view = []
        num_humans_in_view = 0

        for i in range(self.human_num):
            visible = norm(np.array([self.robot.px, self.robot.py]) - np.array([self.humans[i].px, self.humans[i].py])) <= 128
            if visible:
                humans_in_view.append(self.humans[i])
                num_humans_in_view = num_humans_in_view + 1
                human_ids.append(True)
            else:
                human_ids.append(False)

        return humans_in_view, num_humans_in_view, human_ids

    def generate_ob(self, rx, ry):

        ob_ = {}

        visible_humans, num_visibles, human_visibility = self.get_num_human_in_local_map()
        self.update_last_human_states(human_visibility)

        ob_['local_map'] = self.get_local_map(self.semmap, (rx, ry))

        one = torch.ones_like(ob_['local_map'])
        ob_['local_map'] = torch.where(ob_['local_map'] > 0.0, one, ob_['local_map'])

        '''
        plt.subplot(1, 2, 1)
        global_map_vis = visualize_map(self.semmap)
        plt.imshow(global_map_vis)
        plt.scatter(rx, ry, color='g', s=10)
        plt.gca().add_patch(
            plt.Rectangle((rx - 64, ry - 64), 128, 128, fill=False, edgecolor='r', linewidth=2)
            )
        plt.subplot(1, 2, 2)
        local_map_vis = visualize_map(ob_['local_map'])
        plt.imshow(local_map_vis)
        plt.show()
        '''
        t_embedding = self.embedding(self.object_goal_idx)
        ob_['local_map'] = torch.cat([ob_['local_map'], t_embedding], dim = 0)

        ob_['robot_node'] = np.array([self.robot.px, self.robot.py, self.robot.vx, self.robot.vy, self.robot.radius,
                                     self.robot.gx, self.robot.gy, self.robot.v_pref, self.robot.theta])

        #ob = self.compute_observation_for(self.robot)

        ob_['human_node'] = []
        for i in range(self.human_num):
            h = self.last_human_states[i, :]
            ob_['human_node'].append(ObservableState(h[0], h[1], h[2], h[3], h[4]))

        ob_['obstacles'] = self.obstacles

        return ob_

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        speeds = [(np.exp((i + 1) /  4) - 1) / (np.e - 1) * v_pref for i in range(4)]
        rotations = np.linspace(0, 2 * np.pi,  16, endpoint=False)

        #action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        action_space = []
        for j, speed in enumerate(speeds):
            for i, rotation in enumerate(rotations):
                    action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))

        self.action_space_ = action_space

    def reset(self, phase='train', test_case=None):
        time1 = time.clock()
        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0

        if self.kwargs is None:
            self.kwargs = self.precompute_dataset(phase)
            print('kwargs', len(self.kwargs))  # 64
            self.dataset = []
            self.name = []
            for i in range(len(self.kwargs)):
                name_, dataset_ = self.generate_dataset(self.kwargs[i])
                self.dataset.append(dataset_)
                self.name.append(name_)
            self.obstacle_for_allmaps = {}
            for i in range(len(self.dataset)):
                all_maps = self.dataset[i].maps
                for k, v in all_maps.items():
                    Obs = []
                    for j in range(self.obstacle_layers):
                        Obs.append(self.precompute_dataset_for_map(v, j))
                    self.obstacle_for_allmaps[k] = Obs
            print("All maps are generated !!!",len(self.dataset), len(self.name), len(self.obstacle_for_allmaps))

        self.humans = []

        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)

        #print("Loading new environment ...")

        if self.num_per_map % 3 == 0:
            self.env_idx = np.random.randint(0, len(self.kwargs) - 1)
        self.num_per_map += 1
        name, dataset = self.name[self.env_idx], self.dataset[self.env_idx]
        name, self.semmap, self.fmm_dists, _, self.nav_space, self.nav_locs = dataset.get_item_by_name(name)

        self.obstacles = self.obstacle_for_allmaps[name]

        self.occ_map = self.semmap[1:23, ...].sum(axis=0)

        robot_px, robot_py = None, None
        if self.objectnav:
            while robot_px is None:
                robot_px, robot_py, robot_gx, robot_gy = self.generate_robot_for_objectnav_plus()
            #robot_px, robot_py, robot_gx, robot_gy = self.generate_robot_for_objectnav_plus_test()
        else:
            #robot_px, robot_py, robot_gx, robot_gy, path = self.generate_robot()
            robot_px, robot_py, robot_gx, robot_gy = self.generate_robot()
        '''
        plt.subplot(1, 2, 1)
        global_map_vis = visualize_map(self.semmap)
        plt.imshow(global_map_vis)
        plt.subplot(1, 2, 2)
        plt.imshow(self.fmm_dists[self.object_goal_idx])
        plt.show()
        '''
        self.robot.set(robot_px, robot_py, robot_gx, robot_gy, 0, 0, np.pi / 2)

        for _ in range(self.human_num):  # random sample
            self.humans.append(self.generate_human_start())

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()

        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        ob = self.generate_ob(robot_px, robot_py)

        self.states = list()

        self.potential = - self.fmm_dists[self.object_goal_idx, self.robot.py, self.robot.px] * self.resolution
        #self.potential = - norm(np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position())) * self.resolution
        #self.build_action_space(self.v_pref)

        time2 = time.clock()
        #print("Time:", time2 - time1)
        return ob

    def step(self, action, update=True):
        time1 = time.clock()
        #if self.action_space_ is None:
        #    self.build_action_space(self.v_pref)

        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states, self.obstacles)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states, self.obstacles)

        dmin, collision = self.collision_check_between_robot_and_human(action)

        # collision detection between humans
        #if self.collision_check_between_humans():
            # detect collision but don't take humans' collision into account
            # logging.debug('Collision happens between humans in step()')

        # collision detection between robot and obstacle
        collision_with_obstacle = self.collision_check_between_robot_and_obstacles(action)

        rx, ry = self.robot.compute_position(action, self.time_step)

        end_position = np.array([rx, ry])

        #reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) * self.resolution < self.robot.radius

        if self.objectnav:
            reaching_goal = self.success_checker_for_object_nav(end_position, self.success_range)
        else:
            reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) * self.resolution < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
            print("======================", '\033[1;34m Timeout !!! \033[0m', "======================")
        elif collision:
            reward = self.collision_with_human_penalty
            done = True
            info = Collision()
        elif collision_with_obstacle:
            reward = self.collision_with_obstacle_penalty
            done = True
            info = CollisionObs()
            print("======================", '\033[1;32m Collision with obstacle !!! \033[0m', "======================")
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
            #print("======================", '\033[1;31mAchieve Goal Success\033[0m', "======================")
            print("======================", '\033[1;31mAchieve\033[0m', classes_PONI[self.object_goal_idx],
                                                               '\033[1;31mSuccess !!!\033[0m', "======================")
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor #* self.time_step
            done = False
            info = Discomfort(dmin)
            #print("======================", '\033[1;33m Warning: Too close to Humans !!! \033[0m', "======================")
        elif end_position[1] < 20 or end_position[0] < 20 or end_position[0] > self.semmap.shape[2] - 20 \
                or end_position[1] > self.semmap.shape[1] - 20:
            reward = 0
            done = True
            info = Timeout()
            print("======================", '\033[1;34m Timeout (Out of boundry) !!! \033[0m', "======================")
        else:
            potential_cur = - self.fmm_dists[self.object_goal_idx, int(end_position[1]), int(end_position[0])] * self.resolution

            #potential_cur = - norm(end_position - np.array(self.robot.get_goal_position())) * self.resolution

            reward = 0.2 * (- abs(potential_cur) - self.potential)
            self.potential = - abs(potential_cur)

            done = False
            info = Nothing()

        if hasattr(self.robot.policy, 'action_values'):
            self.action_values.append(self.robot.policy.action_values)

        # update all agents
        self.robot.step(action)
        self.humans_step(human_actions)

        ob = self.generate_ob(rx, ry)

        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                            [human.id for human in self.humans]])
        self.global_time += self.time_step

        time2 = time.clock()
        #print("Time:", time2 - time1)
        return ob, reward, done, info

