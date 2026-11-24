
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
import torch
import numpy as np
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent

INRETOUCH_DIR = PROJECT_DIR / "InRetouch"
BENCHMARK_DIR = PROJECT_DIR / "dataset" / "Benchmark"

BASE_CONFIG = (
    INRETOUCH_DIR
    / "options"
    / "train"
    / "InRetouch_Optimize_Single.yml"
)

# PRESETS = ["156"]
PRESETS = ["146", "147","156", "163", "164"]

REFERENCE_IMAGE = "a0081" #53, 81, 89
TEST_IMAGES = [
    "a0001",
    "a0021",
    "a0050",
    "a0045",
    "a0041",
    "a0017",
    "a0020",
]

WORK_DIR = DEMO_DIR / "demo_work"
CONFIG_DIR = WORK_DIR / "configs"
RESULTS_DIR = DEMO_DIR / "results"

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def find_directory(parent, name):
    """Find a directory without depending on capitalization."""

    for path in parent.iterdir():
        if path.is_dir() and path.name.lower() == name.lower():
            return path

    raise FileNotFoundError(
        f"Directory not found: {parent / name}"
    )


def find_image(folder, image_stem):
    """Find an image by stem, allowing optional suffixes via regex."""

    # Match exact stem or stem followed by a non-alphanumeric separator
    # (e.g. a0044_edit, a0044-final, a0044 (1)).
    stem_pattern = re.compile(
        rf"^{re.escape(image_stem)}(?:$|[^0-9A-Za-z].*)",
        flags=re.IGNORECASE,
    )

    matches = [
        p for p in folder.iterdir()
        if p.is_file()
        and stem_pattern.match(p.stem)
    ]

    if not matches:
        raise FileNotFoundError(
            f"Image {image_stem} not found in {folder}"
        )

    return sorted(matches)[0]


def prepare_validation_data(preset):
    """Create small folders containing only the selected test images."""

    preset_dir = f"Preset_{preset}"

    test_root = find_directory(BENCHMARK_DIR, "Test")
    natural_dir = find_directory(test_root, "natural")

    preset_root = find_directory(test_root, "Presets")
    gt_dir = find_directory(preset_root, preset_dir)

    input_stage = WORK_DIR / preset_dir / "input"
    gt_stage = WORK_DIR / preset_dir / "ground_truth"

    input_stage.mkdir(parents=True, exist_ok=True)
    gt_stage.mkdir(parents=True, exist_ok=True)

    for image_name in TEST_IMAGES:
        input_image = find_image(natural_dir, image_name)
        gt_image = find_image(gt_dir, image_name)

        shutil.copy2(input_image, input_stage / input_image.name)
        shutil.copy2(gt_image, gt_stage / gt_image.name)

    return input_stage, gt_stage


def prepare_config(preset, input_stage, gt_stage):
    """Create a separate training configuration for each preset."""

    reference_root = find_directory(
        BENCHMARK_DIR, "Test_References"
    )

    reference_natural = find_directory(
        reference_root, "natural"
    )

    reference_preset_root = find_directory(
        reference_root, "Presets"
    )

    reference_output = find_directory(
        reference_preset_root, f"Preset_{preset}"
    )

    before_image = find_image(
        reference_natural, REFERENCE_IMAGE
    )

    after_image = find_image(
        reference_output, REFERENCE_IMAGE
    )

    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Unique experiment name for each preset
    run_name = f"demo_preset_{preset}"
    config["name"] = run_name

    # Training reference pair
    train = config["datasets"]["train"]

    train["style_natural"] = [str(before_image.resolve())]
    train["style_output"] = [str(after_image.resolve())]

    # Validation images
    val = config["datasets"]["val"]

    val["inp_natural"] = str(input_stage.resolve())
    val["inp_gt"] = str(gt_stage.resolve())

    # Enable PSNR and SSIM
    config["val"]["save_img"] = True

    config["val"]["metrics"] = {
        "psnr": {
            "type": "calculate_psnr",
            "crop_border": 0,
            "test_y_channel": False,
        },
        "ssim": {
            "type": "calculate_ssim",
            "crop_border": 0,
            "test_y_channel": False,
        },
    }

    # Select GPU if available; otherwise CPU
    config["num_gpu"] = 1 if torch.cuda.is_available() else 0

    config["path"]["pretrain_network_g"] = None
    config["path"]["resume_state"] = None

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config_path = CONFIG_DIR / f"{run_name}.yml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return config_path, run_name


# --------------------------------------------------
# Output discovery
# --------------------------------------------------

def find_generated_image(run_name, image_name, start_time):
    """Find the output image generated during this training run."""

    experiment_dir = INRETOUCH_DIR / "experiments" / run_name

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}"
        )

    candidates = []

    # Match exact stem or stem followed by a non-alphanumeric separator
    # (e.g. a0001-jmac_DSC1459_out, a0001_out, a0001 (1)).
    stem_pattern = re.compile(
        rf"^{re.escape(image_name)}(?:$|[^0-9A-Za-z].*)",
        flags=re.IGNORECASE,
    )

    for path in experiment_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        if not stem_pattern.match(path.stem):
            continue

        if path.stat().st_mtime >= start_time:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Generated output for {image_name} "
            f"not found in {experiment_dir}"
        )

    # Prefer files that look like generated outputs over ground-truth copies.
    def candidate_key(path):
        stem = path.stem.lower()
        is_output_like = bool(
            re.search(r"(?:^|[^0-9a-z])(out|output)(?:$|[^0-9a-z])", stem)
        )
        is_gt_like = bool(
            re.search(r"(?:^|[^0-9a-z])(gt|ground[_-]?truth)(?:$|[^0-9a-z])", stem)
        )
        return (
            1 if is_output_like else 0,
            0 if is_gt_like else 1,
            path.stat().st_mtime,
        )

    return max(candidates, key=candidate_key)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_metrics(output_path, gt_path, lpips_model=None, device=None):

    output = np.asarray(
        Image.open(output_path).convert("RGB")
    )

    target = np.asarray(
        Image.open(gt_path).convert("RGB")
    )

    if output.shape != target.shape:
        raise ValueError(
            f"Image dimensions differ: {output.shape}, {target.shape}"
        )

    psnr = peak_signal_noise_ratio(
        target, output, data_range=255
    )

    ssim = structural_similarity(
        target,
        output,
        channel_axis=2,
        data_range=255,
    )

    lpips_value = None

    if lpips_model is not None:
        import torch

        output_tensor = (
            torch.from_numpy(output)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device) / 127.5 - 1
        )

        target_tensor = (
            torch.from_numpy(target)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device) / 127.5 - 1
        )

        with torch.no_grad():
            lpips_value = lpips_model(
                output_tensor, target_tensor
            ).item()

    return psnr, ssim, lpips_value


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)

    if not BENCHMARK_DIR.exists():
        raise FileNotFoundError(BENCHMARK_DIR)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # LPIPS is optional if its package/weights are unavailable.
    lpips_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    except Exception as error:
        print(f"LPIPS unavailable: {error}")
        print("PSNR and SSIM will still be calculated.")

    all_results = []

    for preset in PRESETS:

        print("\n" + "=" * 60)
        print(f"TRAINING PRESET {preset}")
        print("=" * 60)

        input_stage, gt_stage = prepare_validation_data(preset)

        config_path, run_name = prepare_config(
            preset, input_stage, gt_stage
        )

        print(f"Reference image: {REFERENCE_IMAGE}")
        print(f"Training configuration: {config_path}")
        print(f"Device: {device}")

        start_time = time.time()

        # Run the official INRetouch training implementation.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "basicsr.train_INR",
                "-opt",
                str(config_path),
            ],
            cwd=INRETOUCH_DIR,
            check=True,
        )

        print(f"\nTraining completed for preset {preset}.")

        preset_results_dir = RESULTS_DIR / f"Preset_{preset}"
        preset_results_dir.mkdir(parents=True, exist_ok=True)

        for image_name in TEST_IMAGES:

            generated = find_generated_image(
                run_name, image_name, start_time
            )

            gt_image = find_image(gt_stage, image_name)

            saved_output = (
                preset_results_dir / f"{image_name}.png"
            )

            shutil.copy2(generated, saved_output)

            psnr, ssim, lpips_value = calculate_metrics(
                saved_output,
                gt_image,
                lpips_model,
                device,
            )

            result = {
                "preset": preset,
                "image": image_name,
                "PSNR": psnr,
                "SSIM": ssim,
                "LPIPS": lpips_value,
            }

            all_results.append(result)

            print(
                f"{image_name}: "
                f"PSNR={psnr:.4f}, "
                f"SSIM={ssim:.4f}, "
                f"LPIPS={lpips_value if lpips_value is not None else 'N/A'}"
            )

    # Save all metrics
    csv_path = RESULTS_DIR / "metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "preset",
                "image",
                "PSNR",
                "SSIM",
                "LPIPS",
            ],
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Metrics CSV: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
