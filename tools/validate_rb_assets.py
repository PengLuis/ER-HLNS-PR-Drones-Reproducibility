"""Validate the fixed Dujiangyan-derived RB release assets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "data" / "real_maps_final" / "R_DJ_C" / "final"
EXPECTED_HASHES = {
    ASSET_DIR / "r_dj_c_drive_cleaned.graphml": "b3d96518533e33683e3e52b9e0d3cfc74f4836622ec676e3a3d1280927e96779",
    ASSET_DIR / "r_dj_c_poi_candidates.json": "d33df119f8ed562148ece00ee22f858301f9e6a896f6723c122682a453f00918",
    ASSET_DIR / "r_dj_c_fixed_tasks.json": "5c4c4f32f3ea19ae57f94d062a82293a293184c3c628f510f8e17f69323e870a",
    ROOT / "tools" / "prepare_dujiangyan_real_case.py": "ce39f2cbcc710b6a54d3430af04ca0d12ffa36df9566d652144530494fc5647e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"SHA-256 mismatch: {path.relative_to(ROOT)}")

    graph = nx.read_graphml(ASSET_DIR / "r_dj_c_drive_cleaned.graphml")
    if graph.is_directed() or graph.number_of_nodes() != 462 or graph.number_of_edges() != 709:
        raise AssertionError("RB graph must be undirected with 462 nodes and 709 edges")

    task_data = json.loads((ASSET_DIR / "r_dj_c_fixed_tasks.json").read_text(encoding="utf-8"))
    counts = Counter(task["task_class"] for task in task_data["tasks"])
    expected_counts = {"routine_bulk": 8, "time_critical_lightweight": 12}
    if dict(counts) != expected_counts:
        raise AssertionError(f"Unexpected RB task composition: {dict(counts)}")

    print("RB assets verified: 462 nodes, 709 edges, 8 routine tasks, 12 TC tasks.")


if __name__ == "__main__":
    main()
