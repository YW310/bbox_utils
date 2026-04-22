# Demo data for `merge_bboxes.py`

This folder provides a minimal runnable example.

## Files
- `blocks/block_0001.txt`
- `blocks/block_0002.txt`
- `bboxes.json`

## Run
```bash
python merge_bboxes.py \
  --blocks_dir ./demo/blocks \
  --bbox_json ./demo/bboxes.json \
  --out_dir ./demo/output \
  --iou_thr 0.1 \
  --expand_ratio 1.5 \
  --save_vis
```

Expected output:
- `demo/output/merged_bboxes.json`
- `demo/output/crops/merged_0000.txt`, ...
- `demo/output/visualization/merged_bboxes.obj`
- `demo/output/visualization/expanded_bboxes.obj`
