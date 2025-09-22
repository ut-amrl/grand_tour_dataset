import kleinkram
from pathlib import Path
import os

SCRATCH_DIR = os.getenv("SCRATCH")

missions = kleinkram.list_missions(project_names=["GrandTourDataset"])

download_folder = f"{SCRATCH_DIR}/grand_tour_dataset" # Change this to your desired download folder.
patterns = ["*_hdr_front.bag", "*_dlio.bag", "*_anymal_state.bag"] # Regex patterns to match file names or do ["*"] for all files.

Path(download_folder).mkdir(parents=True, exist_ok=True)

vaild_mission_names = []
for m in missions:
    files = kleinkram.list_files(mission_ids=[m.id], file_names=patterns)
    if len(files) == 0:
        print(f"No files found for mission {m.name} matching patterns {patterns}")
        vaild_mission_names.append(m.name)
        continue

    mission_dir = Path(download_folder) / str(m.name)
    mission_dir.mkdir(parents=True, exist_ok=True)
    kleinkram.download(
        file_ids=[f.id for f in files],
        dest=str(mission_dir),
        verbose=True,
        overwrite=True
    )

print(f"Failed mission names: {vaild_mission_names}")