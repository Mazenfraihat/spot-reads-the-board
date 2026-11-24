import argparse
import json
import random
import shutil
from pathlib import Path


def _normalized_polygon(corners, width, height):
    out = []
    for x, y in corners:
        nx = min(max(float(x) / max(width, 1), 0.0), 1.0)
        ny = min(max(float(y) / max(height, 1), 0.0), 1.0)
        out.extend([nx, ny])
    return out


def _write_label_file(label_path, coords):
    values = " ".join(f"{v:.6f}" for v in coords)
    label_path.write_text(f"0 {values}\n", encoding="utf-8")


def _load_samples(synthetic_dir):
    img_dir = Path(synthetic_dir)
    gt_dir = img_dir / "gt"
    samples = []
    for img_path in sorted(img_dir.glob("synth_*.jpg")):
        gt_path = gt_dir / f"{img_path.stem}.json"
        if not gt_path.exists():
            continue
        meta = json.loads(gt_path.read_text(encoding="utf-8"))
        corners = meta.get("corners_tl_tr_br_bl")
        if not corners or len(corners) != 4:
            continue
        samples.append({"image": img_path, "corners": corners})
    return samples


def build_dataset(synthetic_dir, output_root, val_ratio=0.2, seed=42):
    samples = _load_samples(synthetic_dir)
    if not samples:
        raise RuntimeError(f"No synth_*.jpg + gt json samples found in {synthetic_dir}")

    rng = random.Random(seed)
    rng.shuffle(samples)

    n_val = max(1, int(round(len(samples) * val_ratio)))
    val_set = samples[:n_val]
    train_set = samples[n_val:]
    if not train_set:
        train_set, val_set = val_set, train_set

    out_root = Path(output_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_split(items, split):
        for sample in items:
            img_path = sample["image"]
            dst_img = out_root / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            import cv2

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            coords = _normalized_polygon(sample["corners"], w, h)
            label_path = out_root / "labels" / split / f"{img_path.stem}.txt"
            _write_label_file(label_path, coords)

    write_split(train_set, "train")
    write_split(val_set, "val")

    yaml_path = out_root / "whiteboard.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: whiteboard",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"train": len(train_set), "val": len(val_set), "yaml": str(yaml_path)}


def main():
    parser = argparse.ArgumentParser(description="Build YOLO-seg dataset from synthetic whiteboard ground truth.")
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--output-root", default="datasets/whiteboard-seg")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_dataset(
        synthetic_dir=args.synthetic_dir,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        f"Prepared dataset at {args.output_root} "
        f"(train={result['train']}, val={result['val']}).\n"
        f"Data config: {result['yaml']}"
    )


if __name__ == "__main__":
    main()
