# register
from gym.envs.registration import register
from itertools import product

agent_num = range(1,30)
speed = 5
start_ori_array = []
goal_array = []
visu_delay = 0.3
reward_list = {"goal": 100, "collision": -10, "wait": -10, "move": -1}

time_limit = 100
collision = "terminated"

map_list = [
    "map_3x3",
    "map_5x4",
    "map_8x5",
    "map_10x6",
    "map_10x8",
    "map_10x10",
    "map_aoba00",
    "map_aoba01",
]

for an, map_name in product(agent_num, map_list):
    """
    コマンド
    """
    
    register(
        id="drp-{0}agent_{1}-v2".format(an, map_name),
        #entry_point="gym_vrp.envs:VrpEnv",
        entry_point="drp_env.drp_env:DrpEnv",
        kwargs={
            "agent_num": an,
            "speed": speed,
            "start_ori_array": start_ori_array,
            "goal_array": goal_array,
            "visu_delay": visu_delay,
            "time_limit": time_limit,
            "collision": collision,
            "map_name": map_name,
            "reward_list": reward_list,
        },
    )

delay_prob_list = [0.01, 0.05, 0.1, 0.3]
max_delay_list = [1, 2, 3, 5]

# 変動遅延：遅延ステップ数を 1~max_delay の範囲でランダムに決定
for an, map_name, dp, md in product(agent_num, map_list, delay_prob_list, max_delay_list):
    register(
        id="drpDelay-{0}agent_{1}_dp{2}_md{3}-v1".format(an, map_name, dp, md),
        entry_point="drp_env.drp_env:DrpEnv",
        kwargs={
            "agent_num": an,
            "speed": speed,
            "start_ori_array": start_ori_array,
            "goal_array": goal_array,
            "visu_delay": visu_delay,
            "time_limit": time_limit,
            "collision": collision,
            "map_name": map_name,
            "reward_list": reward_list,
            "delay_prob": dp,
            "max_delay": md,
            "delay_fixed": False,
            "delay_seed": None,
        },
    )

# 固定遅延：遅延ステップ数を常に fixed_delay で固定
fixed_delay_list = [1, 2, 3, 5]

for an, map_name, dp, fd in product(agent_num, map_list, delay_prob_list, fixed_delay_list):
    register(
        id="drpDelayFixed-{0}agent_{1}_dp{2}_fd{3}-v1".format(an, map_name, dp, fd),
        entry_point="drp_env.drp_env:DrpEnv",
        kwargs={
            "agent_num": an,
            "speed": speed,
            "start_ori_array": start_ori_array,
            "goal_array": goal_array,
            "visu_delay": visu_delay,
            "time_limit": time_limit,
            "collision": collision,
            "map_name": map_name,
            "reward_list": reward_list,
            "delay_prob": dp,
            "max_delay": fd,
            "delay_fixed": True,
            "delay_seed": None,
        },
    )

neargridmap_list = [
    "map_3x3",
    "map_5x4",
    "map_8x5",
    "map_10x6",
    "map_10x8",
    "map_10x10",
]

state_value = {
    "goal": 10,
    "neighbor": 0.1,
    "around": 0.1,
}

for an, map_name in product(agent_num, map_list):
    register(
        id="drpDT-{0}agent_{1}-v2".format(an, map_name),
        #entry_point="gym_vrp.envs:VrpEnv",
        entry_point="drp_env.DTenv.v2_goalcenter.env_wrapper:DtEnv",
        kwargs={
            "agent_num": an,
            "speed": speed,
            "start_ori_array": start_ori_array,
            "goal_array": goal_array,
            "visu_delay": visu_delay,
            "time_limit": time_limit,
            "collision": collision,
            "map_name": map_name,
            "reward_list": reward_list,
            "state_repre_flag": "DT",
            "state_value": state_value,
        },
    )

for an, map_name in product(agent_num, map_list):
    register(
        id="drpDT-{0}agent_{1}-v1".format(an, map_name),
        #entry_point="gym_vrp.envs:VrpEnv",
        entry_point="drp_env.DTenv.v1_startcenter.env_wrapper:DtEnv",
        kwargs={
            "agent_num": an,
            "speed": speed,
            "start_ori_array": start_ori_array,
            "goal_array": goal_array,
            "visu_delay": visu_delay,
            "time_limit": time_limit,
            "collision": collision,
            "map_name": map_name,
            "reward_list": reward_list,
            "state_repre_flag": "DT",
            "state_value": state_value,
        },
    )



agent_num = 2
speed = 5
start_ori_array = []
goal_array = []
visu_delay = 0.3
reward_list = {"goal": 100, "collision": -10, "wait": -10, "move": -1}

time_limit = 100
collision = "terminated"

map_list = [
    "map_3x3",
    "map_5x4",
    "map_8x5",
    "map_10x6",
    "map_10x8",
    "map_10x10",
    "map_aoba00",
    "map_aoba01",
]

for map_name in map_list:
    register(
        id="drpOA-{0}agent_{1}-v1".format(agent_num, map_name),
        #entry_point="gym_vrp.envs:VrpEnv",
        entry_point="drp_env.OneAgentEnv.env_wrapper:OAEnv",
        kwargs={
            "agent_num": agent_num,
            "speed": speed,
            "start_ori_array": start_ori_array,
            "goal_array": goal_array,
            "visu_delay": visu_delay,
            "time_limit": time_limit,
            "collision": collision,
            "map_name": map_name,
            "reward_list": reward_list,
        },
    )
