from pathlib import Path
import json
import csv


def import_json(file_path:Path):
    print(f"Loading file: {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_files(responses:list, path_json:Path, path_csv:Path):
    """
    This function saves the generated synthetic data into a
    json file and a csv file.

    Args:
        responses: A list with the generated data to store into files.
        path_json: Path for the json file.
        path_csv: Path for the csv file.

    Returns:
        None, save a json file and a csv file.
    """

    print("Saving json file")
    with path_json.open("w") as f:
        json.dump(responses, f, indent=4)
    
    print("Saving csv file")
    keys = responses[0].keys()
    with path_csv.open("w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(responses)
