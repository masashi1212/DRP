import gym
import numpy as np
from drp_env.drp_env import DrpEnv
from drp_env.EE_map import MapMake

class SafeEnv(DrpEnv):
	"""
	エージェント同士が衝突する行動を事前に検知し，joint_actionを安全な行動へ書き換えてからstep()を実行する環境．
	衝突を起こしたエピソードを強制終了させる代わりに，行動を置き換えて学習を継続させる．
	"""

	def step(self, joint_action):

		pre_safe_actions = [int(action) for action in joint_action]
		joint_action = list(pre_safe_actions)  # Safe処理前の行動を残し、別のリストを書き換える

		do = True
		while do:
			do = False
			for i in range(self.agent_num):
				# 他のエージェントと同じ目標ノードへ向かう場合
				if self.current_goal[i] is None: # 自分がノード上にいる時
					for j in range(self.agent_num):
						if j != i and joint_action[i] == joint_action[j]:
							if joint_action[i] != self.current_start[i]:
								joint_action[i] = self.current_start[i]
								do = True  # 条件が変わる可能性があるため，もう一度ループを回す
							break

				# 正面衝突
				if self.current_goal[i] is None: # 自分がノード上にいる時
					for j in range(self.agent_num):
						if (
							j != i 
							and (
								joint_action[j] == self.current_start[i] 
								and joint_action[i] == self.current_start[j]
							)
						):
							joint_action[i] = self.current_start[i]
							do = True
							break

		obs, ri_array, self.terminated, info = super().step(
			joint_action,
			pre_safe_actions=pre_safe_actions,
		)

		return obs, ri_array, self.terminated, info
