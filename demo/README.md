# Demo data for `merge_bboxes.py`

This folder provides a minimal runnable example.

## Files
- `blocks/block_0001.txt`
- `blocks/block_0002.txt`
- `blocks/block_0003.txt`
- `blocks/block_0004.txt`
- `bboxes.json`
- `bboxes/block_0001.json`
- `bboxes/block_0002.json`
- `bboxes/block_0003.json`
- `bboxes/block_0004.json`

## Run
默认参数已指向本 demo（`blocks_dir=./demo/blocks`, `bbox_json=./demo/bboxes`, `out_dir=./demo/output`），可直接运行：

```bash
python merge_bboxes.py --save_vis
```

或显式传参运行：

```bash
python merge_bboxes.py \
  --blocks_dir ./demo/blocks \
  --bbox_json ./demo/bboxes \
  --out_dir ./demo/output \
  --iou_thr 0.1 \
  --expand_ratio 1.5 \
  --save_vis
```

Expected output:
- `demo/output/merged_bboxes.json`
- `demo/output/crops/merged_0000.txt`, ...
- `demo/output/input_crop_bboxes.json`（未 merge 的 bbox，和 input_crops 一一对应）
- `demo/output/input_crops/input_0000.txt`, ...
- `demo/output/visualization/input_bboxes.obj`
- `demo/output/visualization/merged_bboxes.obj`
- `demo/output/visualization/expanded_bboxes.obj`

## Demo highlights
- `bboxes/` 目录使用 `<block_id>.json` 命名（如 `block_0001.json`），与 `blocks/<block_id>.txt` 一一对应。
- label=0 target is duplicated across **3 adjacent blocks** (`block_0001`~`block_0003`) to validate transitive/connected merging.
- label=2 target is duplicated across `block_0002` and `block_0004`.
- label=1 target is duplicated around `(5, 5, 5)` across `block_0003` and `block_0004`.
