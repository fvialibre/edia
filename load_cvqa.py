import os
from pathlib import Path
from datasets import load_dataset, load_from_disk

OUT = Path("data/cvqa_full_dataset")
NPROC = os.cpu_count() or 4

# Optional: faster Hub downloads (requires: pip install hf_transfer)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

if OUT.exists():
    dataset = load_from_disk(str(OUT))
else:
    dataset = load_dataset(
        "afaji/cvqa",
        split="test",
        num_proc=NPROC,
        download_mode="reuse_dataset_if_exists",
    )
    dataset.save_to_disk(
        str(OUT),
        num_proc=NPROC,
        max_shard_size="1GB",
    )