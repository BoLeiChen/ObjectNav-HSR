from HRO_sim.envs.utils.agent import Agent
from HRO_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, observation, masks, rnn_hxs, imitation_learning=False):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        if imitation_learning:
            state = JointState(self.get_full_state(), observation['human_node'])
            action = self.policy.predict(state, observation['obstacles'])
        else:
            state = JointState(self.get_full_state(), observation['human_node'])
            action, rnn_hxs = self.policy.predict(state, masks, rnn_hxs, observation['local_map'])
        return action, rnn_hxs
