import os
import json
import random

def merge_json_files(input_dir, output_file):
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    merged_data = []

    for json_file in json_files:
        print(json_file)
        with open(os.path.join(input_dir, json_file), "r", encoding="utf-8") as file:
            data = json.load(file)

            for item in data:
                merged_data.append(item)
        
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(merged_data, outfile, indent=2, ensure_ascii=False)



