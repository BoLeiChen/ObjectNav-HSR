from gym.envs.registration import register

register(
    id='HROSim-v0',
    entry_point='HRO_sim.envs:HROSim',
)

register(
    id='HROSimDict-v0',
    entry_point='HRO_sim.envs:HROSimDict',
)
