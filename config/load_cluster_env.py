#!/usr/bin/env python3
"""Load config/cluster.yaml and print bash-assignable CLUSTER_* variables.

Usage:
  eval "$(python3 config/load_cluster_env.py [path/to/cluster.yaml])"
"""

from __future__ import annotations

import sys
from pathlib import Path


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        die(
            "PyYAML is required to read cluster config. "
            "Install: pip install pyyaml   or   sudo apt install python3-yaml"
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as e:
        die(f"cannot read {path}: {e}")
    except yaml.YAMLError as e:
        die(f"invalid YAML in {path}: {e}")
    if not isinstance(data, dict):
        die(f"{path}: top level must be a mapping")
    return data


def build_nodes(data: dict) -> str:
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        die("cluster config: 'nodes' must be a non-empty list")
    parts: list[str] = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            die(f"cluster config: nodes[{i}] must be a mapping")
        login = node.get("sshlogin")
        slots = node.get("slots")
        if login is None or str(login) == "":
            die(f"cluster config: nodes[{i}] missing sshlogin")
        try:
            nslots = int(slots)
        except (TypeError, ValueError):
            die(f"cluster config: nodes[{i}] slots must be an integer")
        if nslots < 1:
            die(f"cluster config: nodes[{i}] slots must be >= 1")
        parts.append(f"{nslots}/{login}")
    return ",".join(parts)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "config" / "cluster.yaml"
    if not cfg_path.is_file():
        die(f"cluster config not found: {cfg_path}")

    data = load_yaml(cfg_path)
    nodes = build_nodes(data)

    sync = data.get("log_sync_sec", 15)
    try:
        sync_i = int(sync)
    except (TypeError, ValueError):
        die("cluster config: log_sync_sec must be an integer")

    workdir = data.get("workdir", None)
    if workdir is None or workdir == "":
        workdir_s = ""
    else:
        workdir_s = str(workdir)

    def q(s: str) -> str:
        return "'" + s.replace("'", "'\"'\"'") + "'"

    print(f"CLUSTER_NODES={q(nodes)}")
    print(f"CLUSTER_LOG_SYNC_SEC={sync_i}")
    print(f"CLUSTER_WORKDIR={q(workdir_s)}")


if __name__ == "__main__":
    main()
