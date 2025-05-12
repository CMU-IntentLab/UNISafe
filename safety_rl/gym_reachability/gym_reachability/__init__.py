from gym.envs.registration import register

register(
    id="dubins_car_img-v1",
    entry_point="gym_reachability.gym_reachability.new_envs:DubinsCarOneEnvImg"
)

register(
    id="dubins_car_img_uqonly-v1",
    entry_point="gym_reachability.gym_reachability.new_envs:DubinsCarOneEnvImgUqOnly"
)

register(
    id="dubins_car_img_running-v1",
    entry_point="gym_reachability.gym_reachability.new_envs:DubinsCarOneEnvImgRunning"
)