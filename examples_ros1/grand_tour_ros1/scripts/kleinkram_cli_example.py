import kleinkram
from pathlib import Path

missions = kleinkram.list_missions(project_names=["GrandTourDataset"])

download_folder = "/tmp/grand_tour_data" # Change this to your desired download folder.
patterns = ["*hdr_left.bag", "*_anymal_state.bag"] # Regex patterns to match file names or do ["*"] for all files.

Path(download_folder).mkdir(parents=True, exist_ok=True)

vaild_mission_names = []
for m in missions:
    files = kleinkram.list_files(mission_ids=[m.id], file_names=patterns)
    if len(files) == 0:
        print(f"No files found for mission {m.name} matching patterns {patterns}")
        vaild_mission_names.append(m.name)
        continue

    kleinkram.download(
        file_ids=[f.id for f in files],
        dest=download_folder,
        verbose=True,
        overwrite=True
    )

print(f"Failed mission names: {vaild_mission_names}")