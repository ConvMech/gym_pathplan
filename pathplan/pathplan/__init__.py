from gym.envs.registration import register

register(
    id='pathplan-v0',
    entry_point='pathplan.envs:PathFindingHallwayEnv',
)