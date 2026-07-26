import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class RenderRecorder:
	"""
	render()で描画された各ステップの画像をファイルへ保存するクラス。
	保存先: <リポジトリルート>/MyFiles/results_images/<実行時刻>/episode_XXX/step_YYYY.png
	"""

	def __init__(self, base_dir: Optional[Path] = None):
		repo_root = Path(__file__).resolve().parents[1]
		self.base_dir = base_dir if base_dir is not None else repo_root / "MyFiles" / "results_images"
		self.run_dir = self.base_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		self.episode_dir: Optional[Path] = None

	def begin_episode(self, episode: int) -> None:
		"""
		新しいエピソードの開始を記録する。
		"""
		self.episode_dir = self.run_dir / f"episode_{episode:03d}"

	def frame_path(self, step: int) -> Path:
		"""
		指定ステップの保存先パスを返す(必要ならフォルダを作成する)。
		"""
		self.episode_dir.mkdir(parents=True, exist_ok=True)
		return self.episode_dir / f"step_{step:04d}.png"

	def end_episode(self, should_save: bool = True) -> None:
		"""
		直前のエピソードを確定する。should_saveがFalseならフォルダごと削除する。
		保存要否の判定(衝突が起きたか等)はDrpEnv側が行い、ここには結果だけを渡す。
		"""
		if self.episode_dir is None:
			return

		if not should_save:
			shutil.rmtree(self.episode_dir, ignore_errors=True)

		self.episode_dir = None
