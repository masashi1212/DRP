import gym
import numpy as np
from drp_env.drp_env import DrpEnv
from drp_env.EE_map import MapMake

class SafeEnv(DrpEnv):
	"""
	エージェント同士が衝突する行動を事前に検知し，joint_actionを安全な行動へ書き換えてからstep()を実行する環境．
	衝突を起こしたエピソードを強制終了させる代わりに，行動を置き換えて学習を継続させる．
	safe_mode="occupied_node"では，他エージェントが現在いるノードへの出発も禁止する．
	"""

	def step(self, joint_action):

		pre_safe_actions = [int(action) for action in joint_action]
		joint_action = list(pre_safe_actions)  # Safe処理前の行動を残し、別のリストを書き換える
		# 遅延や予定行動に関係なく、ステップ開始時点の実際のノード占有を使う。
		# エッジ上のエージェントの出発元は、占有ノードとして扱わない。
		current_nodes = (
			[self._get_current_node_if_on_node(i) for i in range(self.agent_num)]
			if self.safe_mode == "occupied_node" else None
		)

		do = True
		while do:
			do = False
			for i in range(self.agent_num):
				if (
					current_nodes is not None # safe_mode="occupied_node"の時
					and self.current_goal[i] is None # 自分がノード上にいる時
					and joint_action[i] != self.current_start[i] # ターゲットノードが現在地ノードでない時
					and any(j != i and node == joint_action[i] for j, node in enumerate(current_nodes)) # 現在地ノードに向かう他のエージェントがいる時
				):
					joint_action[i] = self.current_start[i]
					do = True

				# 他のエージェントと同じ目標ノードへ向かう場合
				if self.current_goal[i] is None: # 自分がノード上にいる時
					for j in range(self.agent_num):
						if j != i and joint_action[i] == joint_action[j]:
							if joint_action[i] != self.current_start[i]:
								joint_action[i] = self.current_start[i]
								do = True  # 条件が変わる可能性があるため，もう一度ループを回す
							break

				# エージェント同士がエッジ上で正面衝突する場合
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
