from typing import Dict, Sequence

import numpy as np

from drp_env.delay_config import DelayConfig


class DelayGenerator:
    """
    DelayConfigに従って、新規遅延の対象エージェントと
    遅延継続時間を生成するクラス。
    """

    def __init__(
        self,
        config: DelayConfig,
        agent_num: int,
    ) -> None:
        if agent_num < 1:
            raise ValueError(f"agent_num must be greater than or equal to 1: {agent_num}")

        self.config = config
        self.agent_num = agent_num
        self.rng = np.random.default_rng(config.seed)

    def sample_new_delays(
        self,
        candidate_ids: Sequence[int],
    ) -> Dict[int, int]:
        """
        遅延抽選対象のエージェントから新規遅延対象を選び、
        エージェントIDと遅延継続時間の辞書を返す。

        Parameters
        ----------
        candidate_ids:
            新規遅延の抽選対象となるエージェントID。
            DrpEnv側で、次の条件を満たすものだけを渡す。

            ・ゴールしていない
            ・ノード上にいる
            ・現在遅延中ではない

        Returns
        -------
        Dict[int, int]
            キーがエージェントID、
            値が今回発生した遅延の継続ステップ数。
        """

        candidates = np.asarray(candidate_ids, dtype=int)

        if candidates.ndim != 1:
            raise ValueError("candidate_ids must be one-dimensional")

        if len(candidates) == 0:
            return {}

        if np.any(candidates < 0) or np.any(candidates >= self.agent_num):
            raise ValueError(f"candidate_ids contains an invalid agent ID: {candidates.tolist()}")

        if len(np.unique(candidates)) != len(candidates):
            raise ValueError(f"candidate_ids must not contain duplicates: {candidates.tolist()}")

        selected_ids = self._sample_event_agents(candidates)

        return {
            int(agent_id): self._sample_duration()
            for agent_id in selected_ids
        }

    def _sample_event_agents(
        self,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        """
        event_modelに従って、新規遅延対象を選択する。
        """

        # ベルヌーイの場合
        if self.config.event_model == "bernoulli":
            occurred = self.rng.random(len(candidate_ids)) < self.config.p_delay

            return candidate_ids[occurred]

        # ポアソンの場合
        poisson_lambda = self.agent_num * self.config.p_delay

        event_num = int(self.rng.poisson(poisson_lambda))

        event_num = min(event_num, len(candidate_ids))

        if event_num == 0:
            return np.empty(0, dtype=int)

        selected_ids = self.rng.choice(candidate_ids, size=event_num, replace=False)

        return np.asarray(selected_ids, dtype=int)

    def _sample_duration(self) -> int:
        """
        duration_modelに従って、遅延継続時間を生成する。
        """

        if self.config.min_delay == self.config.max_delay:
            return self.config.min_delay

        if self.config.duration_model == "uniform":
            return int(self.rng.integers(self.config.min_delay, self.config.max_delay + 1))

        delay_range = self.config.max_delay - self.config.min_delay

        mean = self.config.min_delay + delay_range * self.config.gauss_mean_ratio

        standard_deviation = delay_range * self.config.gauss_std_ratio

        if standard_deviation == 0.0:
            sampled_duration = mean
        else:
            sampled_duration = self.rng.normal(mean, standard_deviation)

        rounded_duration = int(np.floor(sampled_duration + 0.5))

        return int(np.clip(rounded_duration, self.config.min_delay, self.config.max_delay))
