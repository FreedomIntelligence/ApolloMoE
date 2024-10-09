import os
import json
import random
from pathlib import Path

# Source directory and target directory
source_directory = "/path/to/metadata/major_languages/test"
target_directory = "/path/to/metadata/major_languages/dev"

# Get all JSON files in the source directory
json_files = [file for file in os.listdir(source_directory) if file.endswith(".json")]

# Iterate through each JSON file
for json_file in json_files:
    # Build full file path
    file_path = os.path.join(source_directory, json_file)

    # Read lines from JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Calculate the sample size of 5%
    sample_size = int(len(data) * 0.1)

    # Randomly select 5% of the sample
    sampled_data = random.sample(data, sample_size)

    # Build target file path
    target_file_path = os.path.join(target_directory, json_file)

    with open(target_file_path, "w", encoding="utf-8") as f:
        print(target_file_path)
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print("Write the extracted samples to the target file")
