#!/usr/bin/env python3
"""
Merge duplicated 3D AABB detections across point-cloud blocks, then export cropped points.

Input:
- block txt files: x,y,z,sem_label,r,g,b (comma separated, one point per line)
- bbox json list: each item has block_id, center, extent, label, score

Output:
- merged_bboxes.json
- crops/merged_0000.txt, merged_0001.txt, ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class BBoxRecord:
    index: int
    block_id: str
    center: np.ndarray  # (3,)
    extent: np.ndarray  # (3,)
    label: int
    score: float
    min_corner: np.ndarray  # (3,)
    max_corner: np.ndarray  # (3,)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge duplicated 3D bboxes across blocks.")
    parser.add_argument("--blocks_dir", type=Path, required=True, help="Directory containing block txt files")
    parser.add_argument("--bbox_json", type=Path, required=True, help="Path to bbox JSON file")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--iou_thr", type=float, default=0.1, help="IoU threshold for merging")
    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=1.5,
        help="Extent scale factor for crop extraction (default: 1.5)",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="If set, export merged bbox wireframes as OBJ in out_dir/visualization",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.blocks_dir.exists() or not args.blocks_dir.is_dir():
        raise FileNotFoundError(f"blocks_dir does not exist or is not a dir: {args.blocks_dir}")
    if not args.bbox_json.exists():
        raise FileNotFoundError(f"bbox_json does not exist: {args.bbox_json}")
    if args.iou_thr < 0 or args.iou_thr > 1:
        raise ValueError(f"iou_thr must be in [0, 1], got {args.iou_thr}")
    if args.expand_ratio <= 0:
        raise ValueError(f"expand_ratio must be > 0, got {args.expand_ratio}")


def load_bboxes(bbox_json: Path) -> List[BBoxRecord]:
    with bbox_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("bbox_json must contain a list of bbox objects")

    bboxes: List[BBoxRecord] = []
    required_keys = {"block_id", "center", "extent", "label", "score"}

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"bbox item[{idx}] is not an object")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"bbox item[{idx}] missing keys: {sorted(missing)}")

        center = np.asarray(item["center"], dtype=float)
        extent = np.asarray(item["extent"], dtype=float)
        if center.shape != (3,) or extent.shape != (3,):
            raise ValueError(f"bbox item[{idx}] center/extent must be length-3")
        if np.any(extent <= 0):
            raise ValueError(f"bbox item[{idx}] extent must be positive")

        label = int(item["label"])
        if label < 0:
            raise ValueError(f"bbox item[{idx}] label must be >=0")

        min_corner = center - extent / 2.0
        max_corner = center + extent / 2.0

        bboxes.append(
            BBoxRecord(
                index=idx,
                block_id=str(item["block_id"]),
                center=center,
                extent=extent,
                label=label,
                score=float(item["score"]),
                min_corner=min_corner,
                max_corner=max_corner,
            )
        )

    return bboxes


def iou_3d(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_extent = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_extent))
    if inter_vol <= 0:
        return 0.0

    vol_a = float(np.prod(np.maximum(max_a - min_a, 0.0)))
    vol_b = float(np.prod(np.maximum(max_b - min_b, 0.0)))
    union = vol_a + vol_b - inter_vol
    if union <= 0:
        return 0.0
    return inter_vol / union


def merge_bboxes_by_label(bboxes: Sequence[BBoxRecord], iou_thr: float) -> List[dict]:
    merged: List[dict] = []
    labels = sorted({b.label for b in bboxes})

    for label in labels:
        label_boxes = [b for b in bboxes if b.label == label]
        n = len(label_boxes)
        if n == 0:
            continue

        uf = UnionFind(n)
        for i in range(n):
            bi = label_boxes[i]
            for j in range(i + 1, n):
                bj = label_boxes[j]
                if iou_3d(bi.min_corner, bi.max_corner, bj.min_corner, bj.max_corner) > iou_thr:
                    uf.union(i, j)

        groups: Dict[int, List[BBoxRecord]] = {}
        for i, box in enumerate(label_boxes):
            root = uf.find(i)
            groups.setdefault(root, []).append(box)

        for members in groups.values():
            mins = np.stack([m.min_corner for m in members], axis=0)
            maxs = np.stack([m.max_corner for m in members], axis=0)
            merged_min = mins.min(axis=0)
            merged_max = maxs.max(axis=0)
            merged_center = (merged_min + merged_max) / 2.0
            merged_extent = merged_max - merged_min
            merged_score = max(m.score for m in members)

            merged.append(
                {
                    "label": label,
                    "score": float(merged_score),
                    "center": merged_center.tolist(),
                    "extent": merged_extent.tolist(),
                    "min_corner": merged_min.tolist(),
                    "max_corner": merged_max.tolist(),
                    "source_indices": [m.index for m in members],
                    "source_block_ids": sorted({m.block_id for m in members}),
                    "num_merged": len(members),
                }
            )

    return merged


def load_block_points(block_file: Path) -> np.ndarray:
    """Load block points into shape (N, 7): x,y,z,sem_label,r,g,b."""
    try:
        points = np.loadtxt(block_file, delimiter=",", dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to read block file {block_file}: {e}") from e

    if points.size == 0:
        return np.empty((0, 7), dtype=float)

    if points.ndim == 1:
        if points.shape[0] != 7:
            raise ValueError(f"Invalid point format in {block_file}: expected 7 columns")
        points = points[None, :]

    if points.shape[1] != 7:
        raise ValueError(f"Invalid point format in {block_file}: expected 7 columns, got {points.shape[1]}")

    return points


def collect_crop_points(
    merged_bbox: dict,
    blocks_dir: Path,
    expand_ratio: float,
    block_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    center = np.asarray(merged_bbox["center"], dtype=float)
    extent = np.asarray(merged_bbox["extent"], dtype=float)
    expanded_extent = extent * expand_ratio
    crop_min = center - expanded_extent / 2.0
    crop_max = center + expanded_extent / 2.0

    collected = []
    for block_id in merged_bbox["source_block_ids"]:
        if block_id not in block_cache:
            block_file = blocks_dir / f"{block_id}.txt"
            if not block_file.exists():
                raise FileNotFoundError(f"Missing block point file for block_id={block_id}: {block_file}")
            block_cache[block_id] = load_block_points(block_file)

        pts = block_cache[block_id]
        if pts.shape[0] == 0:
            continue

        xyz = pts[:, :3]
        mask = np.all((xyz >= crop_min) & (xyz <= crop_max), axis=1)
        if np.any(mask):
            collected.append(pts[mask])

    if not collected:
        return np.empty((0, 7), dtype=float)

    return np.concatenate(collected, axis=0)


def save_points_txt(points: np.ndarray, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if points.shape[0] == 0:
        out_file.write_text("", encoding="utf-8")
        return

    # sem_label/r/g/b are saved as integers while xyz keeps 6 decimals.
    with out_file.open("w", encoding="utf-8") as f:
        for row in points:
            x, y, z, sem_label, r, g, b = row.tolist()
            f.write(f"{x:.6f},{y:.6f},{z:.6f},{int(round(sem_label))},{int(round(r))},{int(round(g))},{int(round(b))}\n")


def box_corners_from_min_max(min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    """Return the 8 corners of an AABB in a fixed order."""
    x0, y0, z0 = min_corner.tolist()
    x1, y1, z1 = max_corner.tolist()
    return np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )


def save_bboxes_wireframe_obj(merged: Sequence[dict], out_file: Path, use_expanded: bool = False) -> None:
    """
    Save merged bboxes as wireframe OBJ.

    Notes:
    - Each bbox contributes 8 vertices + 12 line segments.
    - If use_expanded=True, it visualizes expanded crop boxes.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 1),
        (5, 6), (6, 7), (7, 8), (8, 5),
        (1, 5), (2, 6), (3, 7), (4, 8),
    ]

    with out_file.open("w", encoding="utf-8") as f:
        f.write("# merged bbox wireframe\n")
        vertex_offset = 0
        for i, m in enumerate(merged):
            center = np.asarray(m["center"], dtype=float)
            extent_key = "expanded_extent" if use_expanded else "extent"
            extent = np.asarray(m[extent_key], dtype=float)
            min_corner = center - extent / 2.0
            max_corner = center + extent / 2.0
            corners = box_corners_from_min_max(min_corner, max_corner)

            f.write(f"o merged_{i:04d}_label_{m['label']}\n")
            for v in corners:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for a, b in edges:
                f.write(f"l {vertex_offset + a} {vertex_offset + b}\n")
            vertex_offset += 8


def run_pipeline(
    blocks_dir: Path,
    bbox_json: Path,
    out_dir: Path,
    iou_thr: float,
    expand_ratio: float,
    save_vis: bool = False,
) -> None:
    bboxes = load_bboxes(bbox_json)
    merged = merge_bboxes_by_label(bboxes, iou_thr=iou_thr)

    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    block_cache: Dict[str, np.ndarray] = {}

    for idx, m in enumerate(merged):
        crop_points = collect_crop_points(
            merged_bbox=m,
            blocks_dir=blocks_dir,
            expand_ratio=expand_ratio,
            block_cache=block_cache,
        )
        crop_file = crops_dir / f"merged_{idx:04d}.txt"
        save_points_txt(crop_points, crop_file)

        m["expanded_extent"] = (np.asarray(m["extent"], dtype=float) * expand_ratio).tolist()
        m["crop_file"] = str(crop_file.relative_to(out_dir).as_posix())
        m["num_crop_points"] = int(crop_points.shape[0])

    merged_json = out_dir / "merged_bboxes.json"
    with merged_json.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    if save_vis:
        vis_dir = out_dir / "visualization"
        save_bboxes_wireframe_obj(merged, vis_dir / "merged_bboxes.obj", use_expanded=False)
        save_bboxes_wireframe_obj(merged, vis_dir / "expanded_bboxes.obj", use_expanded=True)

    print(f"Done. merged boxes: {len(merged)}")
    print(f"Saved: {merged_json}")
    print(f"Crops directory: {crops_dir}")
    if save_vis:
        print(f"Visualization directory: {out_dir / 'visualization'}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_pipeline(
        blocks_dir=args.blocks_dir,
        bbox_json=args.bbox_json,
        out_dir=args.out_dir,
        iou_thr=args.iou_thr,
        expand_ratio=args.expand_ratio,
        save_vis=args.save_vis,
    )


if __name__ == "__main__":
    main()
