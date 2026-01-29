from datasets import load_dataset

CORPORA = [
    "facebook/flores",
    "google/wmt24pp",
    "mteb/NTREX",
]

print("Starting download of 3 parallel corpora...")

for repo_id in CORPORA:
    print(f"Downloading {repo_id}")
    try:
        config = "all" if "flores" in repo_id else None
        load_dataset(repo_id, name=config)
        print(f"Finished {repo_id}")
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")

print("All datasets downloaded.")
