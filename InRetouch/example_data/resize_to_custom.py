import argparse
from pathlib import Path
import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resize_image(src_path: Path, dst_path: Path, scale: float) -> bool:
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dst_path), resized)


def process_folder(input_root: Path, output_root: Path, scale: float) -> None:
    files = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    total = len(files)
    ok = 0
    failed = 0

    for src in files:
        rel = src.relative_to(input_root)
        dst = output_root / rel

        if resize_image(src, dst, scale):
            ok += 1
        else:
            failed += 1
            print(f"[WARN] Failed to process: {src}")

    print("\nDone")
    print(f"Input folder : {input_root}")
    print(f"Output folder: {output_root}")
    print(f"Scale factor : {scale}")
    print(f"Total images : {total}")
    print(f"Success      : {ok}")
    print(f"Failed       : {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce image resolution and save with same internal folder structure."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"D:\UOM\Semester 5\Computer Vision\Project\InRetouch\example_data\Custom_Original"),
        help="Input root folder (default: Custom_Original)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\UOM\Semester 5\Computer Vision\Project\InRetouch\example_data\Custom"),
        help="Output root folder (default: Custom)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor (e.g., 0.5 halves width and height)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input folder does not exist: {args.input}")
    if args.scale <= 0:
        raise ValueError("Scale must be > 0")

    process_folder(args.input, args.output, args.scale)


if __name__ == "__main__":
    main()
