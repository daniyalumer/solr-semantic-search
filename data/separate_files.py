import os
import shutil
from pathlib import Path

def separate_files(base_dir: str):
    cv_dir = os.path.join(base_dir, "CV")
    profile_dir = os.path.join(base_dir, "PROFILE")

    # Create directories if they don't exist
    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(profile_dir, exist_ok=True)

    # Walk through all files in the base directory
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if "CV" in filename:
                target_dir = cv_dir
            elif "PROFILE" in filename:
                target_dir = profile_dir
            else:
                continue  # Skip files that don't match the criteria

            source_path = os.path.join(root, filename)
            target_path = os.path.join(target_dir, filename)

            # Move the file to the target directory
            shutil.move(source_path, target_path)
            print(f"Moved: {source_path} -> {target_path}")

if __name__ == "__main__":
    base_directory = "data/dataset"
    separate_files(base_directory)