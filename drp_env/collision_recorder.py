"""全ステップを一時記録し、一度でも衝突したエピソードだけJSONへ保存する。"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


# EPyMARLを使わない場合の、同一プロセス内で共有する実行ID。
_PROCESS_RUN_ID = uuid4().hex


def _now():
    """日時はタイムゾーン付きUTCで記録し、表示時に現地時刻へ変換できるようにする。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class CollisionRecorder:
    def __init__(self, run_id=None, base_dir=None):
        self.run_id = run_id or _PROCESS_RUN_ID  # 同じ実行に属するログの共通ID。
        self.environment_id = uuid4().hex  # 並列環境間のエピソード番号の重複を区別するID。
        self.base_dir = Path(base_dir) if base_dir is not None else (
            Path(__file__).resolve().parents[1] / "MyFiles" / "collision_logs"
        )  # 作業ディレクトリに依存しない保存先。
        self.episode = None  # 現在のエピソードのメタデータと全ステップ。保存後は解放する。

    def begin_episode(self, env):
        """ランダム生成後の開始・最終ゴールと、実行時のマップを固定して保存する。"""
        self.episode = {
            "schema_version": 1,
            "run_id": self.run_id,
            "environment_id": self.environment_id,
            "process_id": os.getpid(),
            "episode_number": env.episode_account,
            "started_at": _now(),
            "ended_at": None,
            "completed": False,
            "end_reason": None,
            "map_name": env.map_name,
            # 後日CSVが変更されても、当時のマップを再現できるようにする。
            "map": {
                "nodes": [{"id": int(n), "x": float(p[0]), "y": float(p[1])}
                          for n, p in env.pos.items()],
                "edges": [[int(a), int(b)] for a, b in env.G.edges],
            },
            "agent_num": env.agent_num,
            "start_nodes": [int(n) for n in env.start_ori_array],
            "goal_nodes": [int(n) for n in env.goal_array],  # current_goal（経由先）ではない。
            "speed": env.speed,
            "time_limit": env.time_limit,
            "collision_mode": env.collision,
            "collision_distance": 5,  # MapMake.collision_detectの判定条件: 距離 < 5。
            "allow_delay": bool(env.allow_delay),
            "delay_config": asdict(env.delay_config) if env.delay_config is not None else None,
            "collision_events": [],
            "steps": [],
        }
        self.record_step(env, env.obs, {"collision": False, "goal": False, "timeup": False})

    def record_step(self, env, attempted_obs, info):
        if self.episode is None:
            return
        timestamp = self.episode["started_at"] if env.step_account == 0 else _now()
        # 全衝突ペアを記録する。条件は既存のcollision_detectと同じで、判定自体は変更しない。
        pairs = []
        if info["collision"]:
            pairs = [[i, j] for i in range(env.agent_num) for j in range(i + 1, env.agent_num)
                     if sum((float(attempted_obs[i][k]) - float(attempted_obs[j][k])) ** 2
                            for k in (0, 1)) < self.episode["collision_distance"] ** 2]
            self.episode["collision_events"].append({
                "step": env.step_account, "timestamp": timestamp, "pairs": pairs,
            })
        self.episode["steps"].append({
            "step": env.step_account,
            "timestamp": timestamp,
            "collision": bool(info["collision"]),
            "collision_pairs": pairs,
            "goal": bool(info["goal"]),
            "timeup": bool(info["timeup"]),
            "agents": [{
                "id": i,
                "position": [float(env.obs[i][k]) for k in (0, 1)],
                # 衝突時に取り消された移動先。これを使うと実際の衝突判定地点を描画できる。
                "collision_position": [float(attempted_obs[i][k]) for k in (0, 1)]
                                      if info["collision"] else None,
                "is_delayed": bool(env.is_delays[i]),  # このステップで遅延が適用されたか。
                "delay_remaining": int(env.delay_remaining[i]),  # 今回分を消費した後の残り。
                # terminatedは衝突・時間切れでもTrueになるので、座標でゴール到達を判定する。
                "goal_reached": all(float(env.obs[i][k]) == float(env.pos[env.goal_array[i]][k])
                                    for k in (0, 1)),
                "terminated": bool(env.terminated[i]),
	                # 行動がSafe処理と遅延処理でどう変化したかを段階別に残す。
	                "pre_safe_action": env.pre_safe_actions[i] if env.pre_safe_actions is not None else None,
	                "post_safe_action": env.post_safe_actions[i] if env.post_safe_actions is not None else None,
	                "post_delay_action": env.post_delay_actions[i] if env.post_delay_actions is not None else None,
	                # 既存ログ利用者との互換性のため、従来の2項目も残す。
                "planned_action": env.planned_actions[i] if env.planned_actions is not None else None,
                "executed_action": env.executed_actions[i] if env.executed_actions is not None else None,
            } for i in range(env.agent_num)],
        })

    def end_episode(self, reason, completed=False):
        """衝突なしは破棄。途中のreset/closeではcompleted=Falseで保存し、二重保存を防ぐ。"""
        if self.episode is None:
            return
        if self.episode["collision_events"]:
            self.episode["ended_at"] = self.episode["steps"][-1]["timestamp"] if completed else _now()
            self.episode["completed"] = completed
            self.episode["end_reason"] = reason
            # 実験ごとのフォルダに、エピソード番号を8桁にゼロ埋めして保存する。
            run_dir = self.base_dir / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            target = run_dir / f"{self.episode['episode_number']:08d}.json"
            temporary = target.with_suffix(".json.tmp")  # 書き込み途中のログをビューアに読ませない。
            temporary.write_text(json.dumps(self.episode, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(target)
        self.episode = None
