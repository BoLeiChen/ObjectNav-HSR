import numpy as np
import rvo2
from HRO_sim.envs.policy.policy import Policy
from HRO_sim.envs.utils.action import ActionXY
from HRO_sim.envs.utils.state import ObservableState_nov, JointState

class ORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 128   #10
        self.max_neighbors = 128   #10
        self.time_horizon = 25     #5
        self.time_horizon_obst = 25 #5
        self.radius = 0.15    #0.3 for no obstacles
        self.max_speed = 2
        self.sim = None
        self.add_wall = True
        self.add_object = True
        self.point_cnt = None

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state, obstacle = None):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        cnt = 0
        for i in range(len(obstacle)):
            cnt += len(obstacle[i])
        robot_state = state.robot_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and (self.sim.getNumAgents() != len(state.human_states) + 1 or cnt != self.point_cnt):
            del self.sim
            self.sim = None
        self.point_cnt = cnt

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius / 0.05, self.max_speed)
            self.sim.addAgent(robot_state.position, *params, (robot_state.radius + 0.01 + self.safety_space) / 0.05,
                              robot_state.v_pref, robot_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, (human_state.radius + 0.01 + self.safety_space) / 0.05,
                                  self.max_speed, human_state.velocity)

            for level in range(len(obstacle)):
                if level==0:
                    if self.add_wall:
                        for i in range(len(obstacle[level])):#这里是同类型障碍物的个数
                            for j in range(len(obstacle[level][i]) - 1):#j是一个障碍物的点的个数
                                self.sim.addObstacle([(obstacle[level][i][j,0,0], obstacle[level][i][j,0,1]), (obstacle[level][i][j+1,0,0], obstacle[level][i][j+1,0,1])])
                else:
                    if self.add_object:
                        for i in range(len(obstacle[level])):#这里是同类型障碍物的个数
                            for j in range(len(obstacle[level][i]) - 1):#j是一个障碍物的点的个数
                                self.sim.addObstacle([(obstacle[level][i][j,0,0], obstacle[level][i][j,0,1]), (obstacle[level][i][j+1,0,0], obstacle[level][i][j+1,0,1])])
            self.sim.processObstacles()

        else:
            self.sim.setAgentPosition(0, robot_state.position)
            self.sim.setAgentVelocity(0, robot_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((robot_state.gx - robot_state.px, robot_state.gy - robot_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))

        self.last_state = state

        return action

#行人用这个来选择动作
class CentralizedORCA(ORCA):
    def __init__(self):
        super().__init__()

    def predict(self, state, obstacle = None):
        """ Centralized planning for all agents """
        cnt = 0
        for i in range(len(obstacle)):
            cnt += len(obstacle[i])
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and (self.sim.getNumAgents() != len(state) or cnt != self.point_cnt):
            del self.sim
            self.sim = None
        self.point_cnt = cnt

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius / 0.05, self.max_speed)
            for agent_state in state:
                self.sim.addAgent(agent_state.position, *params, (agent_state.radius + 0.01 + self.safety_space) / 0.05,
                                  self.max_speed, agent_state.velocity)

            for level in range(len(obstacle)):
                if level==0:
                    if self.add_wall:
                        for i in range(len(obstacle[level])):#这里是同类型障碍物的个数
                            for j in range(len(obstacle[level][i]) - 1):#j是一个障碍物的点的个数
                                self.sim.addObstacle([(obstacle[level][i][j,0,0], obstacle[level][i][j,0,1]), (obstacle[level][i][j+1,0,0], obstacle[level][i][j+1,0,1])])
                else:
                    if self.add_object:
                        for i in range(len(obstacle[level])):#这里是同类型障碍物的个数
                            for j in range(len(obstacle[level][i]) - 1):#j是一个障碍物的点的个数
                                self.sim.addObstacle([(obstacle[level][i][j,0,0], obstacle[level][i][j,0,1]), (obstacle[level][i][j+1,0,0], obstacle[level][i][j+1,0,1])])
            self.sim.processObstacles()

        else:
            for i, agent_state in enumerate(state):
                self.sim.setAgentPosition(i, agent_state.position)
                self.sim.setAgentVelocity(i, agent_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        for i, agent_state in enumerate(state):
            velocity = np.array((agent_state.gx - agent_state.px, agent_state.gy - agent_state.py))
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed if speed > 1 else velocity
            self.sim.setAgentPrefVelocity(i, (pref_vel[0], pref_vel[1]))

        self.sim.doStep()
        actions = [ActionXY(*self.sim.getAgentVelocity(i)) for i in range(len(state))]

        return actions