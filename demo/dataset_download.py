from huggingface_hub import HfApi, snapshot_download

repo = "omaralezaby/Retouch_Transfer_Dataset"
dataset = r"D:\UOM\Semester 5\Computer Vision\Project\dataset"

files = HfApi().list_repo_files(repo, repo_type="dataset")

selected = [
    f for f in files
    if f.startswith("Benchmark/Test/Presets/")
    # if f.startswith("Benchmark/Test_Reference/Presets/")
    and any(f"/Preset_{p}/" in f for p in ["146", "147"])
    and "a0001" <= f.split("/")[-1][:5] <= "a0117"
    and f.endswith((".jpg", ".png", ".jpeg"))
]

snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=dataset,
    allow_patterns=selected
)