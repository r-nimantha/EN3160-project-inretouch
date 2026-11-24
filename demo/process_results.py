import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / "results" / "metrics.csv")
df["preset"] = df["preset"].astype(str)

output_dir = base_dir / "processed_results"
output_dir.mkdir(exist_ok=True)

metrics = ["PSNR", "SSIM", "LPIPS"]

summary = df.groupby("preset")[metrics].agg(["mean", "std"])
summary.columns = ["_".join(col) for col in summary.columns]
summary.to_csv(output_dir / "preset_summary.csv")

overall_stats = df[metrics].agg(["mean", "std"])

print("\n=== Overall Statistics (All Samples) ===")
print(overall_stats.round(4))
overall_stats.to_csv(output_dir / "overall_summary.csv")

print("\n=== Preset Summary ===")
print(summary.round(4))

print(f"\nResults saved to: {output_dir}")