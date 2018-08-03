from gym.envs.registration import register

register(
    id='CustomTaxi-v0',
    entry_point='customEnv.taxi:CustomTaxiEnv',
)
