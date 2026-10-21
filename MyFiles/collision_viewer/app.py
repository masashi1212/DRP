"""衝突ログを閲覧するローカルサーバー（Python標準ライブラリのみ）。

起動: python MyFiles/collision_viewer/app.py
閲覧: http://127.0.0.1:8765
"""

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]  # cwdによらずリポジトリを特定する。
VIEWER_DIR = Path(__file__).resolve().parent  # 配信するHTMLの場所。


def read_map(map_dir, map_name):
    """CSVの先頭行はヘッダー。エッジは重複と向きを取り除いて無向に扱う。"""
    folder = (map_dir / map_name).resolve()
    if folder.parent != map_dir.resolve():
        raise ValueError("マップ名が不正です。")
    with (folder / "node.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = csv.reader(f)
        next(rows)
        nodes = [{"id": int(r[0]), "x": round(float(r[1]), 2), "y": round(float(r[2]), 2)}
                 for r in rows if r and any(v.strip() for v in r)]
    with (folder / "edge.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = csv.reader(f)
        next(rows)
        edges = sorted({tuple(sorted((int(r[0]), int(r[1]))))
                        for r in rows if r and any(v.strip() for v in r)})
    node_ids = {n["id"] for n in nodes}
    if any(a not in node_ids or b not in node_ids for a, b in edges):
        raise ValueError("edge.csvに存在しないノードが指定されています。")
    return {"nodes": nodes, "edges": [list(e) for e in edges]}


def map_signature(data):
    """CSVと記録時の地図を、行順やエッジの向きに依存せず比較する。"""
    return (sorted((n["id"], n["x"], n["y"]) for n in data["nodes"]),
            sorted({tuple(sorted(e)) for e in data["edges"]}))


def load_episode(log_dir, map_dir, relative_path):
    target = (log_dir / relative_path).resolve()  # APIから参照するログの絶対パス。
    if log_dir.resolve() not in target.parents or target.suffix != ".json":
        raise ValueError("ログのパスが不正です。")
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not data.get("steps"):
        raise ValueError("未対応のログ形式、またはステップのないログです。")
    try:
        csv_map = read_map(map_dir, data["map_name"])  # 指定されたnode.csv/edge.csvを読み込む。
        if map_signature(csv_map) == map_signature(data["map"]):
            data["display_map"] = csv_map
            data["map_source"] = "node.csv / edge.csv"
        else:
            data["display_map"] = data["map"]
            data["map_source"] = "CSVが記録時と異なるため、ログ内のマップを使用"
    except (OSError, ValueError, StopIteration):
        data["display_map"] = data["map"]
        data["map_source"] = "CSVを読み込めないため、ログ内のマップを使用"
    return data


def make_handler(log_dir, map_dir):
    """指定フォルダのJSONとビューアHTMLだけを配信する。"""
    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200, content_type="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path)
            try:
                if route.path in ("/", "/index.html"):
                    self.reply((VIEWER_DIR / "index.html").read_bytes(),
                               content_type="text/html; charset=utf-8")
                elif route.path == "/api/logs":
                    logs = []  # エピソード選択用の概要一覧。
                    skipped = 0  # 読み込めなかったJSONの数。
                    for path in sorted(log_dir.rglob("*.json"), reverse=True):
                        if log_dir.resolve() not in path.resolve().parents:
                            continue
                        try:
                            data = json.loads(path.read_text(encoding="utf-8"))
                            if data.get("schema_version") != 1 or not data.get("collision_events"):
                                continue
                            logs.append({
                                "path": path.relative_to(log_dir).as_posix(),
                                "run_id": data["run_id"], "map_name": data["map_name"],
                                "episode_number": data["episode_number"], "started_at": data["started_at"],
                                "step_count": len(data["steps"]) - 1,
                            })
                        except (OSError, ValueError, KeyError, TypeError):
                            skipped += 1
                    logs.sort(key=lambda entry: entry["started_at"], reverse=True)
                    self.reply({"logs": logs, "skipped": skipped})
                elif route.path == "/api/episode":
                    params = parse_qs(route.query)
                    self.reply(load_episode(log_dir, map_dir, params.get("path", [""])[0]))
                else:
                    self.reply({"error": "ページが見つかりません。"}, status=404)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.reply({"error": str(exc)}, status=400)

        def log_message(self, *_args):
            pass  # コマ送り中に端末がHTTPログで埋まらないようにする。

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "MyFiles" / "collision_logs")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(args.log_dir.resolve(), REPO_ROOT / "drp_env" / "map"))
    print(f"衝突ログ: {args.log_dir.resolve()}\nブラウザで開く: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
