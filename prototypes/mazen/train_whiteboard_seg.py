import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 segmentation model for whiteboard detection.")
    parser.add_argument("--data", default="datasets/whiteboard-seg/whiteboard.yaml")
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as err:
        raise SystemExit(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from err

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Data config not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        degrees=8,
        translate=0.1,
        scale=0.6,
        shear=2,
        perspective=0.0015,
        hsv_h=0.01,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.2,
    )


if __name__ == "__main__":
    main()
