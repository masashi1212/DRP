# Drone Routing Problems

## 言語 / Language

* [日本語](#日本語)
* [English](#english)

---

## 日本語

## 目次

* [プロジェクト概要](#プロジェクト概要)
* [事前準備](#事前準備)
* [インストール](#インストール)
* [環境の使い方](#環境の使い方)
* [ファイル構成](#ファイル構成)
* [Epymarlとの連携](#epymarlとの連携)

## プロジェクト概要

マルチエージェント強化学習のための配送経路問題（DRP）環境です。[epymarl](https://github.com/uoe-agents/epymarl) に対応しています。

## 事前準備

以下のツールが必要です。インストールされていない場合は先に準備してください。

### ターミナル（コマンドライン）

* **Mac**: アプリケーション > ユーティリティ > `ターミナル` を開く。bash/zsh は最初から使えます。
* **Windows**: `コマンドプロンプト` または `PowerShell` を使用。より使いやすくするには [Git for Windows](https://gitforwindows.org/) をインストールすると Git Bash が使えます。

### Anaconda（Python環境管理ツール）

PythonのバージョンやライブラリをまとめてPC管理するためのツールです。

1. インストーラーをダウンロード: [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. インストーラーを実行して指示に従う
3. インストール後、新しいターミナルを開いて以下を実行:

```bash
conda --version
```

バージョン番号（例: `conda 23.x.x`）が表示されれば成功です。

### Git

このリポジトリをダウンロードするために必要です。

* **Mac**: ターミナルで `git --version` を実行。未インストールの場合は macOS が自動でインストールを促してくれます。
* **Windows**: [https://git-scm.com/](https://git-scm.com/) からダウンロード。

## インストール

### 1. リポジトリをダウンロード

ターミナルを開いて以下を実行:

```bash
git clone https://github.com/Yamaguchi-yushi/MARL4DRP.git
cd MARL4DRP
```

### 2. conda環境を作成・有効化

```bash
conda create -n env_name python=3.9
conda activate env_name
```

> `env_name` は好きな名前に変えてください（例: `drp_env`）。
> 有効化後はターミナルの先頭に `(env_name)` と表示されます。

### 3. 依存ライブラリを一括インストール

`install.sh` があるディレクトリ（リポジトリのルート）で以下を実行:

```bash
bash install.sh
```

> **注意:** `conda-libmamba-solver` に関する警告は無視して構いません。
> `bash` は Mac / Linux では最初から使えます。Windows の場合は Git Bash または WSL を使用してください。

## 環境の使い方

gymフレームワークを使った環境の作成例:

```python
import gym
import drp_env
env = gym.make("drp-2agent_map_3x3-v2", state_repre_flag="onehot_fov")
```

または

```python
import gym
env = gym.make("drp_env:drp-2agent_map_3x3-v2", state_repre_flag="onehot_fov")
```

### 環境名の形式

```
drp-{agent_num}agent_{map_name}-v2
```

* agent_num: エージェント数（1〜6）
* map_name: `map_3x3` / `map_5x4` / `map_8x5` / `map_10x6` / `map_10x8` / `map_10x10` / `map_aoba00` / `map_aoba01`
* state_repre_flag: 観測の種類 — `coordinate` / `onehot` / `onehot_fov` / `heu_onehot` / `heu_onehot_fov`

### 行動（Action）

ノード番号を指定します。無効な行動を選んだ場合、エージェントはその場に止まります。

### 観測（Observation）

...

### 報酬（Reward）

報酬はエージェントごとに設定され、`reward_list` によって決まります。

デフォルト値:

```yaml
reward_list:
  goal: 100
  collision: -10
  wait: -10
  move: -1
```

* **goal**: ゴールに到達したとき `reward = reward_list["goal"]`
* **collision**: 衝突したとき `reward = reward_list["collision"] * speed（デフォルト: 5）`
* **wait**: ゴール以外の場所で停止したとき `reward = reward_list["wait"] * speed`
* **move**: 移動したとき `reward = reward_list["move"] * speed`

### Info（情報辞書）

* **distance_from_start** (List of float): 各エージェントのスタートからの移動距離。停止・衝突時は `speed` 分加算。
* **goal** (Bool): 全エージェントがゴールに到達したとき `True`
* **collision** (Bool): 衝突発生時 `True`
* **timeup** (Bool): タイムリミット到達時 `True`
* **cost** (int): エピソードのコスト。低いほど効率的。成功時は全エージェントの到着ステップ数の合計、失敗時は `agent_num × time_limit`。
* **goal_cost** (int or None): ゴール達成時のみ `cost` と同じ値、失敗時は `None`。

## ファイル構成

<pre>
MARL4DRP
├── README.md
├── install.sh
├── requirements.txt
├── setup.py
├── drp_env
│   ├── __init__.py
│   ├── drp_env.py
│   ├── EE_map.py
│   ├── map
│   └── state_repre
├── drpload_test.py
├── for_epymarl
└── epymarl
</pre>

ファイル・ディレクトリの説明:

名前                              |  説明
----------------------------------|------------------------------------------------------------------------------------
drp_env                           |  drp_env パッケージのディレクトリ
drpload_test.py                   |  drp_env の使用サンプル
for_epymarl                       |  epymarl との連携に必要なファイル
epymarl                           |  マルチエージェントRLフレームワーク（epymarl v1.0.0）
install.sh                        |  依存ライブラリの一括インストールスクリプト

drp_env 内のファイル:

名前                              |  説明
----------------------------------|------------------------------------------------------------------------------------
\_\_init\_\_.py                   |  環境の登録
drp_env.py                        |  gymフレームワークに準拠した環境
EE_map.py                         |  ネットワーク構造の処理
map                               |  マップ情報のCSVファイル
state_repre                       |  観測の管理

## Epymarlとの連携

このリポジトリの `epymarl` フォルダには、`drp_env` に対応した設定済みの epymarl が含まれています。

`epymarl` ディレクトリに移動してから学習を実行:

```bash
cd epymarl
python3 src/main.py --config=iql --env-config=gymma with env_args.time_limit=100 'env_args.key=drp_env:drp-1agent_map_3x3-v2' env_args.state_repre_flag="onehot"
```

> **注意:** `:` を含む引数はシェルの解析エラーを避けるためシングルクォート `'` で囲んでください。

### 使用可能なアルゴリズム

`--config=iql` の部分を以下に変更して使用できます:

Config  | アルゴリズム
--------|----------
`iql`   | Independent Q-Learning
`qmix`  | QMIX
`vdn`   | Value Decomposition Networks

---

## English

### Table of Contents

* [About the Project](#about-the-project)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Environment](#environment)
* [File Structure](#file-structure)
* [Using Epymarl](#using-epymarl)

### About the Project

A multi-agent reinforcement learning environment for drone routing problems (DRP), compatible with [epymarl](https://github.com/uoe-agents/epymarl).

### Prerequisites

Before starting, you need the following tools installed on your machine.

#### Terminal (command line)

* **Mac**: Open `Terminal` from Applications > Utilities. bash/zsh is available by default — no installation needed.
* **Windows**: Use `Command Prompt` or `PowerShell`. For a better experience, consider installing [Git for Windows](https://gitforwindows.org/) which includes Git Bash.

#### Anaconda (Python environment manager)

Anaconda is required to manage Python versions and packages.

1. Download the installer from: [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. Run the installer and follow the on-screen instructions
3. After installation, open a new terminal and verify it works:

```bash
conda --version
```

If you see a version number (e.g. `conda 23.x.x`), the installation was successful.

#### Git (en)

Git is required to download this repository.

* **Mac**: Run `git --version` in the terminal. If not installed, macOS will prompt you to install it automatically.
* **Windows**: Download from [https://git-scm.com/](https://git-scm.com/)

### Installation

#### 1. Download this repository

Open a terminal and run:

```bash
git clone https://github.com/Yamaguchi-yushi/MARL4DRP.git
cd MARL4DRP
```

#### 2. Create and activate a conda environment

```bash
conda create -n env_name python=3.9
conda activate env_name
```

> Replace `env_name` with any name you like (e.g. `drp_env`).
> After activation, you should see `(env_name)` at the beginning of your terminal prompt.

#### 3. Install all dependencies

Make sure you are in the root directory of this repository (the folder containing `install.sh`), then run:

```bash
bash install.sh
```

> **Note:** Warnings about `conda-libmamba-solver` can be safely ignored.
> `bash` is available by default on Mac and Linux. On Windows, use Git Bash or WSL.

### Environment

How to create environments with the gym framework:

```python
import gym
import drp_env
env = gym.make("drp-2agent_map_3x3-v2", state_repre_flag="onehot_fov")
```

or

```python
import gym
env = gym.make("drp_env:drp-2agent_map_3x3-v2", state_repre_flag="onehot_fov")
```

#### Environment name

```text
drp-{agent_num}agent_{map_name}-v2
```

* agent_num: number of agents, 1~6
* map_name: `map_3x3` / `map_5x4` / `map_8x5` / `map_10x6` / `map_10x8` / `map_10x10` / `map_aoba00` / `map_aoba01`
* state_repre_flag: kind of observation — `coordinate` / `onehot` / `onehot_fov` / `heu_onehot` / `heu_onehot_fov`

#### Action

Node number. When taking an invalid action, the agent stops at its current position.

#### Observation

...

#### Reward

Rewards are set per agent, determined by `reward_list`.

Default:

```yaml
reward_list:
  goal: 100
  collision: -10
  wait: -10
  move: -1
```

* **goal**: When an agent reaches its goal, `reward = reward_list["goal"]`
* **collision**: When agents collide, `reward = reward_list["collision"] * speed (default: 5)`
* **wait**: When an agent stops at a non-goal position, `reward = reward_list["wait"] * speed (default: 5)`
* **move**: When an agent moves, `reward = reward_list["move"] * speed (default: 5)`

#### Info

* **distance_from_start** (List of float): Distance traveled from start node for each agent. Increases by `speed` when an agent stops or collides.
* **goal** (Bool): `True` when all agents have reached their goals.
* **collision** (Bool): `True` when agents collide.
* **timeup** (Bool): `True` when the time limit is reached (used by epymarl).
* **cost** (int): Episode cost. Lower is better. Equals the sum of arrival steps on success, or `agent_num * time_limit` on failure.
* **goal_cost** (int or None): Same as `cost` when all agents reach their goals; `None` otherwise.

### File Structure

```text
MARL4DRP
├── README.md
├── install.sh
├── requirements.txt
├── setup.py
├── drp_env
│   ├── __init__.py
│   ├── drp_env.py
│   ├── EE_map.py
│   ├── map
│   └── state_repre
├── drpload_test.py
├── for_epymarl
└── epymarl
```

Name                              |  Description
----------------------------------|------------------------------------------------------------------------------------
drp\_env                          |  the directory for package drp\_env
drpload\_test.py                  |  a sample file using drp\_env
for\_epymarl                      |  files required to work with epymarl
epymarl                           |  multi-agent RL framework (epymarl v1.0.0)
install.sh                        |  installation script for all dependencies

Directories/files in drp\_env:

Name                              |  Description
----------------------------------|------------------------------------------------------------------------------------
\_\_init\_\_.py                   |  register environments
drp\_env.py                       |  environment with gym structure
EE\_map.py                        |  process related to network structure
map                               |  csv files about map information
state\_repre                      |  manage observation of environments

### Using Epymarl

The `epymarl` folder in this repository already contains the modified version of epymarl configured to work with `drp_env`.

Run training from the `epymarl` directory:

```bash
cd epymarl
python3 src/main.py --config=iql --env-config=gymma with env_args.time_limit=100 'env_args.key=drp_env:drp-1agent_map_3x3-v2' env_args.state_repre_flag="onehot"
```

> **Note:** Use single quotes `'` around arguments containing `:` to avoid shell parsing issues.

#### Available algorithms

Replace `--config=iql` with any of the following:

Config  | Algorithm
--------|----------
`iql`   | Independent Q-Learning
`qmix`  | QMIX
`vdn`   | Value Decomposition Networks
