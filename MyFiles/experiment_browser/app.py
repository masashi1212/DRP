"""epymarl/results/sacred と epymarl/results/tb_logs を unique_token で紐づけて
実験条件から絞り込んだり、unique_token から実験条件を逆引きするための Streamlit アプリ。

このファイル(および loader.py)は DRP リポジトリの外に置いても動作する。
DRP リポジトリのルート(epymarl/ を含むディレクトリ)の絶対パスを、
コマンドライン引数 --repo-root かアプリ内の入力欄のどちらかで指定すること。

起動方法:
    streamlit run app.py -- --repo-root "/path/to/DRP"
    (--repo-root を省略した場合はアプリ内で入力できる)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from loader import (
    Experiment,
    guess_repo_root,
    is_valid_repo_root,
    load_experiments,
    sacred_root_of,
    tb_logs_root_of,
)

# このスクリプト自身の隣に、最後に指定したリポジトリの場所を保存しておく(再入力の手間を省くため)
CONFIG_PATH = Path(__file__).resolve().parent / ".repo_root.txt"

# env_args のうち、専用フィールドとして既に表示している(=汎用フィルタの対象から除く)キー
KNOWN_ENV_ARGS_KEYS = {
    "key",
    "allow_delay",
    "delay_prob",
    "max_delay",
    "delay_fixed",
    "time_limit",
    "reward_list.goal",
    "reward_list.collision",
    "reward_list.wait",
    "reward_list.move",
}

FIELD_LABELS = {
    "algo": "アルゴリズム",
    "map_name": "マップ名",
    "agent_num": "エージェント数",
    "allow_delay": "実行遅延の有無",
    "delay_prob": "遅延確率",
    "max_delay": "一度遅延した際の遅延ステップ数",
    "delay_fixed": "遅延ステップ数が固定か",
    "time_limit": "1エピソードあたりの最大ステップ数",
    "t_max": "学習させた総ステップ数",
    "reward_goal": "報酬: ゴール到達",
    "reward_collision": "報酬: 衝突",
    "reward_wait": "報酬: 待機",
    "reward_move": "報酬: 移動",
    "seed": "シード値",
    "run_date": "実行日(絞り込み用)",
    "run_datetime": "実行日時",
    "info_json_size_bytes": "info.jsonのファイルサイズ",
    "sacred_rel_path": "sacredでの場所",
    "env_key": "環境ID (env_args.key)",
    "state_repre_flag": "状態表現 (state_repre_flag)",
    "pretrained_wrapper": "pretrained_wrapper",
    "delay_seed": "遅延の乱数シード (delay_seed)",
}

NONE_LABEL = "(値なし)"

st.set_page_config(page_title="実験ブラウザ (sacred x tb_logs)", layout="wide")


def parse_cli_repo_root() -> Optional[str]:
    """`streamlit run app.py -- --repo-root /path/to/DRP` の形式で渡された値を取得する。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", dest="repo_root", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.repo_root


def candidate_repo_roots(raw_input: str) -> list[Path]:
    """入力欄の文字列から、試すべき絶対パス候補を優先順に作る。
    先頭の "/" が抜けている(例: "Users/masashi/.../DRP")場合に備えて、
    "/" を補った候補もあわせて試す。
    """
    raw_input = raw_input.strip()
    raw_candidates = [raw_input]
    if raw_input and not raw_input.startswith(("/", "~")):
        raw_candidates.append("/" + raw_input)

    resolved: list[Path] = []
    seen = set()
    for c in raw_candidates:
        p = Path(c).expanduser().resolve()
        if p not in seen:
            seen.add(p)
            resolved.append(p)
    return resolved


def load_saved_repo_root() -> Optional[str]:
    if CONFIG_PATH.exists():
        text = CONFIG_PATH.read_text().strip()
        return text or None
    return None


def save_repo_root(path_str: str) -> None:
    CONFIG_PATH.write_text(path_str.strip())


def resolve_repo_root() -> Optional[Path]:
    """DRP リポジトリのルートを、以下の優先順位で決定する。
    1. アプリ内で今回のセッション中に入力された値
    2. --repo-root コマンドライン引数
    3. 前回保存された値(このスクリプトと同じフォルダの .repo_root.txt)
    4. このスクリプトの設置場所からの推測(MyFiles/experiment_browser/ に置かれたままの場合のみ有効)
    """
    override = st.session_state.get("repo_root_override")
    if override:
        return Path(override).expanduser().resolve()

    cli_value = parse_cli_repo_root()
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    saved = load_saved_repo_root()
    if saved:
        return Path(saved).expanduser().resolve()

    return guess_repo_root()


@st.cache_data
def _load(sacred_root: Path, tb_logs_root: Path) -> list[Experiment]:
    return load_experiments(sacred_root, tb_logs_root)


def label_for(key: str) -> str:
    return FIELD_LABELS.get(key, key)


def format_datetime(dt) -> str:
    if dt is None:
        return NONE_LABEL
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_size(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return NONE_LABEL
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def infer_kind(values: list[Any]) -> str:
    non_none = [v for v in values if v is not None]
    if not non_none:
        return "empty"
    if all(isinstance(v, bool) for v in non_none):
        return "bool"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_none):
        return "numeric"
    return "categorical"


def numeric_filter(label: str, values: list[Optional[float]], key: str):
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None
    lo, hi = min(non_none), max(non_none)
    if lo == hi:
        st.caption(f"**{label}**: {lo} (全件で同一の値のため絞り込み対象外)")
        return None
    selected = st.slider(label, min_value=float(lo), max_value=float(hi), value=(float(lo), float(hi)), key=key)
    return selected


def bool_or_categorical_filter(label: str, values: list[Any], key: str):
    uniq = sorted({v for v in values}, key=lambda x: (x is None, str(x)))
    if len(uniq) <= 1:
        only = uniq[0] if uniq else None
        st.caption(f"**{label}**: {only if only is not None else NONE_LABEL} (全件で同一の値のため絞り込み対象外)")
        return None
    display = [NONE_LABEL if v is None else str(v) for v in uniq]
    mapping = dict(zip(display, uniq))
    selected_display = st.multiselect(label, display, default=display, key=key)
    return {mapping[d] for d in selected_display}


def apply_numeric(experiments: list[Experiment], getter, rng):
    if rng is None:
        return experiments
    lo, hi = rng
    return [e for e in experiments if (v := getter(e)) is None or (lo <= v <= hi)]


def apply_categorical(experiments: list[Experiment], getter, selected):
    if selected is None:
        return experiments
    return [e for e in experiments if getter(e) in selected]


def build_extra_env_args_keys(experiments: list[Experiment]) -> list[str]:
    keys: set[str] = set()
    for e in experiments:
        keys.update(e.env_args_flat.keys())
    return sorted(keys - KNOWN_ENV_ARGS_KEYS)


def to_row(e: Experiment) -> dict:
    row = {
        "unique_token": e.unique_token,
        "アルゴリズム": e.algo,
        "マップ名": e.map_name,
        "エージェント数": e.agent_num,
        "実行遅延の有無": e.allow_delay,
        "遅延確率": e.delay_prob,
        "遅延ステップ数": e.max_delay,
        "遅延ステップ数固定": e.delay_fixed,
        "最大ステップ数/episode": e.time_limit,
        "総学習ステップ数": e.t_max,
        "報酬:goal": e.reward_goal,
        "報酬:collision": e.reward_collision,
        "報酬:wait": e.reward_wait,
        "報酬:move": e.reward_move,
        "seed": e.seed,
        "実行日時": format_datetime(e.run_datetime),
        "info.jsonサイズ": format_size(e.info_json_size_bytes),
        "tb_logsあり": e.has_tb_logs,
        "sacredでの場所": e.sacred_rel_path,
    }
    for k, v in e.env_args_flat.items():
        if k in KNOWN_ENV_ARGS_KEYS:
            continue
        row[f"env_args.{k}"] = v
    return row


def show_detail(e: Experiment) -> None:
    st.markdown(f"**unique_token**: `{e.unique_token}`")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 実験条件")
        rows = [
            (label_for("algo"), e.algo),
            (label_for("map_name"), e.map_name),
            (label_for("agent_num"), e.agent_num),
            (label_for("allow_delay"), e.allow_delay),
            (label_for("delay_prob"), e.delay_prob),
            (label_for("max_delay"), e.max_delay),
            (label_for("delay_fixed"), e.delay_fixed),
            (label_for("time_limit"), e.time_limit),
            (label_for("t_max"), e.t_max),
            (label_for("reward_goal"), e.reward_goal),
            (label_for("reward_collision"), e.reward_collision),
            (label_for("reward_wait"), e.reward_wait),
            (label_for("reward_move"), e.reward_move),
            (label_for("seed"), e.seed),
            (label_for("run_datetime"), format_datetime(e.run_datetime)),
            (label_for("info_json_size_bytes"), format_size(e.info_json_size_bytes)),
            (label_for("sacred_rel_path"), e.sacred_rel_path),
            (label_for("env_key"), e.env_key),
        ]
        st.table(
            pd.DataFrame(
                [(k, NONE_LABEL if v is None else str(v)) for k, v in rows],
                columns=["項目", "値"],
            ).set_index("項目")
        )
    with col2:
        st.markdown("##### env_args (その他含む全項目)")
        st.json(e.env_args_flat, expanded=False)

        st.markdown("##### tb_logs")
        if e.has_tb_logs:
            st.success(f"見つかりました: `{e.tb_logs_dir}`")
            st.code(f"tensorboard --logdir '{e.tb_logs_dir}'", language="bash")
        else:
            st.warning(f"tb_logs フォルダが見つかりません(想定パス: `{e.tb_logs_dir}`)")
        st.caption(f"sacred: `{e.sacred_rel_path}`")

    with st.expander("config.json 全体を表示"):
        st.json(e.raw_config, expanded=False)


def main() -> None:
    title_col, reload_col = st.columns([6, 1])
    with title_col:
        st.title("実験ブラウザ: sacred ⇔ tb_logs")
    with reload_col:
        st.write("")
        if st.button("🔄 再読み込み", help="新しく実行した実験の結果を反映します"):
            _load.clear()
            st.rerun()
    st.caption(
        "info.json に unique_token が記録されている実験のみが対象です"
        "(unique_token が無い過去の実験は対象外)。"
        "新しい実験を実行した後は「再読み込み」を押してください。"
    )

    repo_root = resolve_repo_root()
    valid = repo_root is not None and is_valid_repo_root(repo_root)

    with st.expander("📁 DRPリポジトリの場所", expanded=not valid):
        st.caption(
            "epymarl/results/sacred と epymarl/results/tb_logs を含む DRP リポジトリの"
            "絶対パスを指定してください。起動時に --repo-root 引数で渡すこともできます。"
        )
        default_value = str(repo_root) if repo_root is not None else ""
        input_value = st.text_input("DRPリポジトリの絶対パス", value=default_value, key="repo_root_input")
        if st.button("この場所を使う"):
            candidates = candidate_repo_roots(input_value)
            match = next((c for c in candidates if is_valid_repo_root(c)), None)
            if match is not None:
                st.session_state["repo_root_override"] = str(match)
                save_repo_root(str(match))
                _load.clear()
                st.rerun()
            else:
                tried = "\n".join(f"- `{c}`" for c in candidates)
                st.error(
                    "epymarl/results/sacred が見つかりませんでした。"
                    f"以下を試しました:\n{tried}"
                )

    if not valid:
        if repo_root is None:
            st.warning("DRPリポジトリの場所が未指定です。上の欄に絶対パスを入力してください。")
        else:
            st.warning(
                f"指定された場所(`{repo_root}`)に epymarl/results/sacred が見つかりません。"
                "上の欄で正しい場所を指定してください。"
            )
        return

    st.caption(f"参照中のリポジトリ: `{repo_root}`")

    sacred_root = sacred_root_of(repo_root)
    tb_logs_root = tb_logs_root_of(repo_root)

    experiments = _load(sacred_root, tb_logs_root)
    if not experiments:
        st.error(
            "unique_token を持つ実験が見つかりませんでした。"
            "epymarl/results/sacred 以下に info.json (unique_token 記録済み) があるか確認してください。"
        )
        return

    with_tb = sum(e.has_tb_logs for e in experiments)
    st.caption(f"対象実験数: {len(experiments)} 件 (うち tb_logs 紐付け成功: {with_tb} 件)")

    tab_filter, tab_lookup = st.tabs(["条件で絞り込む", "unique_token から逆引きする"])

    with tab_filter:
        st.subheader("実験条件で絞り込み")

        with st.expander("フィルタ条件", expanded=True):
            f_algo = bool_or_categorical_filter(label_for("algo"), [e.algo for e in experiments], "f_algo")
            f_map = bool_or_categorical_filter(label_for("map_name"), [e.map_name for e in experiments], "f_map")
            f_agents = numeric_filter(label_for("agent_num"), [e.agent_num for e in experiments], "f_agents")
            f_delay_on = bool_or_categorical_filter(
                label_for("allow_delay"), [e.allow_delay for e in experiments], "f_delay_on"
            )
            f_delay_prob = numeric_filter(label_for("delay_prob"), [e.delay_prob for e in experiments], "f_delay_prob")
            f_max_delay = numeric_filter(label_for("max_delay"), [e.max_delay for e in experiments], "f_max_delay")
            f_delay_fixed = bool_or_categorical_filter(
                label_for("delay_fixed"), [e.delay_fixed for e in experiments], "f_delay_fixed"
            )
            f_time_limit = numeric_filter(label_for("time_limit"), [e.time_limit for e in experiments], "f_time_limit")
            f_t_max = numeric_filter(label_for("t_max"), [e.t_max for e in experiments], "f_t_max")
            f_reward_goal = numeric_filter(label_for("reward_goal"), [e.reward_goal for e in experiments], "f_r_goal")
            f_reward_collision = numeric_filter(
                label_for("reward_collision"), [e.reward_collision for e in experiments], "f_r_collision"
            )
            f_reward_wait = numeric_filter(label_for("reward_wait"), [e.reward_wait for e in experiments], "f_r_wait")
            f_reward_move = numeric_filter(label_for("reward_move"), [e.reward_move for e in experiments], "f_r_move")
            f_seed = numeric_filter(label_for("seed"), [e.seed for e in experiments], "f_seed")
            f_run_date = bool_or_categorical_filter(
                label_for("run_date"), [e.run_date for e in experiments], "f_run_date"
            )

            extra_filters = {}
            extra_keys = build_extra_env_args_keys(experiments)
            if extra_keys:
                st.markdown("###### env_args のその他の項目")
                for k in extra_keys:
                    values = [e.env_args_flat.get(k) for e in experiments]
                    kind = infer_kind(values)
                    if kind == "numeric":
                        extra_filters[k] = ("numeric", numeric_filter(label_for(k), values, f"f_extra_{k}"))
                    elif kind in ("bool", "categorical"):
                        extra_filters[k] = (
                            "categorical",
                            bool_or_categorical_filter(label_for(k), values, f"f_extra_{k}"),
                        )

        filtered = experiments
        filtered = apply_categorical(filtered, lambda e: e.algo, f_algo)
        filtered = apply_categorical(filtered, lambda e: e.map_name, f_map)
        filtered = apply_numeric(filtered, lambda e: e.agent_num, f_agents)
        filtered = apply_categorical(filtered, lambda e: e.allow_delay, f_delay_on)
        filtered = apply_numeric(filtered, lambda e: e.delay_prob, f_delay_prob)
        filtered = apply_numeric(filtered, lambda e: e.max_delay, f_max_delay)
        filtered = apply_categorical(filtered, lambda e: e.delay_fixed, f_delay_fixed)
        filtered = apply_numeric(filtered, lambda e: e.time_limit, f_time_limit)
        filtered = apply_numeric(filtered, lambda e: e.t_max, f_t_max)
        filtered = apply_numeric(filtered, lambda e: e.reward_goal, f_reward_goal)
        filtered = apply_numeric(filtered, lambda e: e.reward_collision, f_reward_collision)
        filtered = apply_numeric(filtered, lambda e: e.reward_wait, f_reward_wait)
        filtered = apply_numeric(filtered, lambda e: e.reward_move, f_reward_move)
        filtered = apply_numeric(filtered, lambda e: e.seed, f_seed)
        filtered = apply_categorical(filtered, lambda e: e.run_date, f_run_date)
        for k, (kind, val) in extra_filters.items():
            getter = (lambda e, k=k: e.env_args_flat.get(k))
            if kind == "numeric":
                filtered = apply_numeric(filtered, getter, val)
            else:
                filtered = apply_categorical(filtered, getter, val)

        st.markdown(f"**該当する実験: {len(filtered)} 件**")

        if filtered:
            df = pd.DataFrame([to_row(e) for e in filtered])
            st.dataframe(df, width="stretch", hide_index=True)

            st.markdown("##### unique_token 一覧 (コピー用)")
            st.code("\n".join(e.unique_token for e in filtered), language="text")

            chosen = st.selectbox(
                "詳細を見る実験を選択",
                options=[e.unique_token for e in filtered],
                key="filter_detail_select",
            )
            if chosen:
                show_detail(next(e for e in filtered if e.unique_token == chosen))
        else:
            st.info("条件に一致する実験がありません。")

    with tab_lookup:
        st.subheader("unique_token から実験条件を調べる")
        tokens = sorted((e.unique_token for e in experiments), reverse=True)
        chosen = st.selectbox("unique_token を選択(入力して絞り込み可)", options=tokens, key="lookup_select")
        if chosen:
            show_detail(next(e for e in experiments if e.unique_token == chosen))


if __name__ == "__main__":
    main()
