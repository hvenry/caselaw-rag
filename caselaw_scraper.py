import os
import requests
import zipfile
import json
from io import BytesIO

"""
Case Law Scraper

This is used to get relevant case information from the Case Law Access Project, which offers free public
access to 6.9 million unique cases of U.S. law.

Obviously we cannot use all of this rich information with our project constraints, but we can specify a
subset of information from: https://static.case.law/.

In this current instance, this will scan a specified amount of volumes from https://static.case.law/ny3d/.

notable urls:
/ny (New York Reports 1800-1997, volumes 1-309):
- New York court of appeals decisions, contains cases decided by the highest court in New York State.

/ny-2d (New York Reports 1956-2003, volumes 1-100),
/ny3d (New York Reports 2003-2017, volumes 1-29):
- New York reports, 2nd and 3rd series, official series of published decisions from the NY Court of Appeals.  

/ny-sup-ct (Supreme Court Reports 1873-1895, volumes 8-99):
- New York Supreme Court decisions, the trial level court of general jurisdiction in NY.
"""


BASE_URL = "https://static.case.law/ny3d/"
OUTPUT_DIR = "data/caselaw_data"
DATA_DIR = "data"
DATASET_FILE = os.path.join(DATA_DIR, "caselaw_dataset.json")

# Specify the range of volumes
VOLUME_START = 1
VOLUME_END = 5

# Make directories if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def download_and_extract_volume(volume_number):
    volume_url = f"{BASE_URL}{volume_number}.zip"
    print(f"Downloading volume {volume_number}...")
    response = requests.get(volume_url)
    if response.status_code != 200:
        print(f"Failed to download volume {volume_number}. Skipping.")
        return None

    # Extract the zip content
    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
        volume_dir = os.path.join(OUTPUT_DIR, f"volume_{volume_number}")
        os.makedirs(volume_dir, exist_ok=True)
        zip_file.extractall(volume_dir)
        return volume_dir


def parse_volume(volume_dir):
    dataset = []

    # Paths to the folders
    json_dir = os.path.join(volume_dir, "json")
    metadata_dir = os.path.join(volume_dir, "metadata")

    # Parse case metadata
    case_metadata_file = os.path.join(metadata_dir, "CasesMetadata.json")
    if os.path.exists(case_metadata_file):
        with open(case_metadata_file, "r", encoding="utf-8") as f:
            case_metadata = json.load(f)
    else:
        case_metadata = {}

    # Parse individual cases
    for json_file in os.listdir(json_dir):
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            case_data = json.load(f)
            case_id = os.path.splitext(json_file)[0]

            # Add metadata if available
            if case_id in case_metadata:
                case_data["metadata"] = case_metadata[case_id]

            dataset.append(case_data)

    return dataset


def main():
    all_data = []

    # Iterate over volume range
    for volume_number in range(VOLUME_START, VOLUME_END):
        volume_dir = download_and_extract_volume(volume_number)
        if volume_dir:
            volume_data = parse_volume(volume_dir)
            all_data.extend(volume_data)

    # Save all data into a single JSON file
    print(f"Saving dataset to {DATASET_FILE}...")

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4)

    print(f"Dataset saved: {DATASET_FILE}")


if __name__ == "__main__":
    main()
