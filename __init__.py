from gym.envs.registration import register

register(
    id='mytaxi-v0',
    entry_point='myenv.mytaxi:MyTaxiEnv'
)


register(
    id='mytaxi-v2',
    entry_point='myenv.mytaxi_stage2:MyTaxiEnv'
)
