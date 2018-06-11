from gym.envs.registration import register

register(
    id='mytaxi-v0',
    entry_point='myenv.mytaxi:MyTaxiEnv'
)
