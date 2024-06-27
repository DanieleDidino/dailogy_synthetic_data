from pathlib import Path
import json


def import_json(file_path:Path):
    print(f"Loading file: {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
