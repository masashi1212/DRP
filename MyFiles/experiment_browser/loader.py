"""sacred (info.json / config.json) と tb_logs を unique_token で紐づけて読み込むモジュール。

対象は info.json に unique_token が記録されている実験のみ(それ以前の実験は対象外)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

def guess_repo_root() -> Optional[Path]:
    """このファイルが DRP リポジトリ直下から2階層下(MyFiles/experiment_browser/)に
    置かれている場合にのみ有効な、あくまでベストエフォートの推測。
    ファイルを別の場所に移動した場合は None を返す(呼び出し側で明示的な指定が必要)。
    """
    guess = Path(__file__).resolve().parents[2]
    return guess if sacred_root_of(guess).is_dir() else None


def sacred_root_of(repo_root) -> Path:
    return Path(repo_root) / "epymarl" / "results" / "sacred"


def tb_logs_root_of(repo_root) -> Path:
    return Path(repo_root) / "epymarl" / "results" / "tb_logs"


def is_valid_repo_root(repo_root) -> bool:
    return sacred_root_of(repo_root).is_dir()

# 例: "drp_env:drpDelay-5agent_map_8x5-v1" / "drp_env:drp-2agent_map_3x3-v2"
ENV_ID_PATTERN = re.compile(
    r"^(?:[^:]+:)?(?P<prefix>[A-Za-z]+)-(?P<agents>\d+)agent_(?P<map>.+)-v(?P<version>\d+)$"
)

# drp_env/__init__.py の map_list と一致させる。マップ名の後ろに旧形式の
# "_dp0.1_fd3" のような余分なサフィックスが付いていても正しくマップ名を切り出すために使う。
MAP_NAMES = [
    "map_3x3",
    "map_5x4",
    "map_8x5",
    "map_10x6",
    "map_10x8",
    "map_10x10",
    "map_aoba00",
    "map_aoba01",
]

# unique_token の末尾に付与される実行日時。例: "..._2026-07-14 14:14:15.366908"
UNIQUE_TOKEN_TIMESTAMP_PATTERN = re.compile(
    r"_(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)$"
)

# config.json / env_args の値が文字列 "None" / "null" として保存されていることがあるための正規化
_NONE_LIKE_STRINGS = {"None", "null"}


def parse_run_datetime(unique_token: str) -> Optional[datetime]:
    """unique_token 末尾の実行日時を取り出す。"""
    m = UNIQUE_TOKEN_TIMESTAMP_PATTERN.search(unique_token)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def _normalize(value: Any) -> Any:
    if isinstance(value, str) and value in _NONE_LIKE_STRINGS:
        return None
    return value


def parse_env_key(env_key: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """env_args.key (gymの環境id) からマップ名とエージェント数を抽出する。"""
    if not env_key:
        return None, None
    m = ENV_ID_PATTERN.match(env_key)
    if not m:
        return None, None
    raw_map = m.group("map")
    # 既知のマップ名で前方一致させ、旧形式の "_dp0.1_fd3" 等の余分なサフィックスを切り落とす
    map_name = next((name for name in MAP_NAMES if raw_map == name or raw_map.startswith(name + "_")), raw_map)
    return map_name, int(m.group("agents"))


def flatten(d: dict, prefix: str = "") -> dict:
    """ネストした辞書 (reward_list など) をドット区切りキーに展開する。"""
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, prefix=f"{key}."))
        else:
            flat[key] = _normalize(v)
    return flat


@dataclass
class Experiment:
    unique_token: str
    sacred_dir: Path
    sacred_rel_path: str
    tb_logs_dir: Path
    has_tb_logs: bool

    algo: Optional[str]
    seed: Optional[int]
    t_max: Optional[int]

    run_date: Optional[date]
    run_datetime: Optional[datetime]
    info_json_size_bytes: int

    env_key: Optional[str]
    map_name: Optional[str]
    agent_num: Optional[int]

    allow_delay: Optional[bool]
    delay_prob: Optional[float]
    max_delay: Optional[int]
    delay_fixed: Optional[bool]

    time_limit: Optional[int]

    reward_goal: Optional[float]
    reward_collision: Optional[float]
    reward_wait: Optional[float]
    reward_move: Optional[float]

    # env_args の全キー(フラット化済み)。フィルタUIやconfig全体の表示に使う。
    env_args_flat: dict = field(default_factory=dict)
    raw_config: dict = field(default_factory=dict)


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_experiments(sacred_root: Path, tb_logs_root: Path) -> list[Experiment]:
    """sacred_root / tb_logs_root は呼び出し側が明示的に指定する
    (例: DRP リポジトリのルートを起点に sacred_root_of() / tb_logs_root_of() で組み立てる)。
    このモジュール自体の設置場所には依存しない。
    """
    experiments: list[Experiment] = []

    sacred_root = Path(sacred_root).resolve()
    tb_logs_root = Path(tb_logs_root)
    # sacred_root は "<repo_root>/epymarl/results/sacred" という前提の下、相対パス表示のために逆算する
    repo_root = sacred_root.parents[2]

    if not sacred_root.exists():
        return experiments

    for info_path in sorted(sacred_root.rglob("info.json")):
        run_dir = info_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        info = _load_json(info_path)
        config = _load_json(config_path)
        if info is None or config is None:
            continue

        unique_token = info.get("unique_token")
        if not unique_token:
            # unique_token が記録されていない古い実験は対象外
            continue

        raw_env_args = config.get("env_args") or {}
        env_args_flat = flatten(raw_env_args)

        env_key = env_args_flat.get("key")
        map_name, agent_num = parse_env_key(env_key)

        tb_logs_dir = tb_logs_root / unique_token
        run_datetime = parse_run_datetime(unique_token)

        try:
            sacred_rel_path = str(run_dir.relative_to(repo_root))
        except ValueError:
            sacred_rel_path = str(run_dir)

        experiments.append(
            Experiment(
                unique_token=unique_token,
                sacred_dir=run_dir,
                sacred_rel_path=sacred_rel_path,
                tb_logs_dir=tb_logs_dir,
                has_tb_logs=tb_logs_dir.is_dir(),
                algo=config.get("name"),
                seed=config.get("seed"),
                t_max=config.get("t_max"),
                run_date=run_datetime.date() if run_datetime else None,
                run_datetime=run_datetime,
                info_json_size_bytes=info_path.stat().st_size,
                env_key=env_key,
                map_name=map_name,
                agent_num=agent_num,
                allow_delay=env_args_flat.get("allow_delay"),
                delay_prob=env_args_flat.get("delay_prob"),
                max_delay=env_args_flat.get("max_delay"),
                delay_fixed=env_args_flat.get("delay_fixed"),
                time_limit=env_args_flat.get("time_limit"),
                reward_goal=env_args_flat.get("reward_list.goal"),
                reward_collision=env_args_flat.get("reward_list.collision"),
                reward_wait=env_args_flat.get("reward_list.wait"),
                reward_move=env_args_flat.get("reward_list.move"),
                env_args_flat=env_args_flat,
                raw_config=config,
            )
        )

    return experiments
