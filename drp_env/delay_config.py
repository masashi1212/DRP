from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DelayConfig:
    """
    実行遅延に関する設定を保持するクラス。
    """

    event_model: str
    """遅延イベントの発生モデル。bernoulli または poisson。"""
    p_delay: float
    """遅延イベントの発生確率に関する値。"""
    duration_model: str
    """遅延継続時間の生成モデル。uniform または gaussian。"""
    min_delay: int
    """遅延継続時間の最小ステップ数。"""
    max_delay: int
    """遅延継続時間の最大ステップ数。"""
    seed: Optional[int] = None
    """遅延生成用の乱数シード。Noneの場合は固定しない。"""
    gauss_mean_ratio: Optional[float] = None
    """Gaussian分布の平均位置を、遅延時間の範囲に対する比率で指定する。duration_model='gaussian'の場合は必須、'uniform'の場合はNoneである必要がある。"""
    gauss_std_ratio: Optional[float] = None
    """Gaussian分布の標準偏差を、遅延時間の範囲に対する比率で指定する。duration_model='gaussian'の場合は必須、'uniform'の場合はNoneである必要がある。"""

    def __post_init__(self) -> None:
        """
        インスタンス生成時に設定値を検証する。
        """

        if self.event_model not in {"bernoulli", "poisson"}:
            raise ValueError(f"event_model must be 'bernoulli' or 'poisson': {self.event_model}")
        
        if self.duration_model not in {"uniform", "gaussian"}:
            raise ValueError(f"duration_model must be 'uniform' or 'gaussian': {self.duration_model}")
        
        if not 0.0 <= self.p_delay <= 1.0:
            raise ValueError(f"p_delay must be between 0.0 and 1.0: {self.p_delay}")
        
        if self.min_delay < 1:
            raise ValueError(f"min_delay must be greater than or equal to 1: {self.min_delay}")
        
        if self.max_delay < self.min_delay:
            raise ValueError(f"max_delay must be greater than or equal to min_delay: min_delay={self.min_delay}, max_delay={self.max_delay}")
        
        if self.duration_model == "uniform":
            if self.gauss_mean_ratio is not None:
                raise ValueError(f"gauss_mean_ratio must be None when duration_model='uniform': {self.gauss_mean_ratio}")

            if self.gauss_std_ratio is not None:
                raise ValueError(f"gauss_std_ratio must be None when duration_model='uniform': {self.gauss_std_ratio}")

        else:  # duration_model == "gaussian"
            if self.gauss_mean_ratio is None:
                raise ValueError("gauss_mean_ratio must be specified when duration_model='gaussian'")

            if not 0.0 <= self.gauss_mean_ratio <= 1.0:
                raise ValueError(f"gauss_mean_ratio must be between 0.0 and 1.0: {self.gauss_mean_ratio}")

            if self.gauss_std_ratio is None:
                raise ValueError("gauss_std_ratio must be specified when duration_model='gaussian'")

            if self.gauss_std_ratio < 0.0:
                raise ValueError(f"gauss_std_ratio must be greater than or equal to 0.0: {self.gauss_std_ratio}")

        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError(f"seed must be int or None: {type(self.seed).__name__}")
