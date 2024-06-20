import environ
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import ollama

from utils.issues_category import issues

# Set constanst
LLM_MODEL = "dolphin-mistral"

# How many synthetic sentences generate for each issue
N_PAIRS = 5


def call_model(prompt: str, system_message: str, model_name: str) -> tuple[str, str]:
    """
    Take a prompt and a model name and use the ollama framework to 
    generate a response.
    It returns a tuple with the model name and the response.
    """

    response = ollama.generate(
            model=model_name,
            prompt=prompt,
            system=system_message)
        
    return response["response"]


def generate_data(issues: list, n_pairs: int) -> list:
    
    # List to store the reposnes
    responses_json = []

    # Count the iteration throught the "issues" list
    N_issue = 1

    for issue in issues:

        # Print the current issue number
        print(f"Generating output {N_issue} of {len(issues)}")
        N_issue += 1

        system_message = """
        You are an AI assistant that outputs only JSON data.
        Do not include any text before or after the JSON response.
        """

        prompt_1 = f"""
        Generate examples of dysfunctional and toxic language that might be encountered between couples or 
        ex-couples who have to continuously interact.
    
        Each entry should include:
    
        1 - A sentence reflecting dysfunctional communication, showcasing various forms of toxicity such as
            insults, harassment, threats, manipulation, and derogatory remarks.
        
        2 - A transformed version of the same sentence that represents functional, healthy communication.
    
        Ensure the sentences are realistic and diverse in terms of content and context.
        The sentences should refer to this issue category:
        '{issue}'
    
        Provide {n_pairs} pairs of sentences.
        """
    
        # The prompt is divided into 2 sub-prompts because
        # the example of the output format uses Curly brackets {},
        # and this cannot be done in a f-string.
        prompt_2 = """
        Always respond only with valid JSON format and nothing else.
        Do not include any text before or after the JSON.

        You must provide the output exactly in the following format:
        
        [
            {
                "dysfunctional": "write here the dysfunctional text",
                "functional": "write here the functional text"
            },
            {
                "dysfunctional": "write here the dysfunctional text",
                "functional": "write here the functional text"
            },
        ]
        """
    
        prompt = prompt_1 + prompt_2

        max_iteration = 3
        num_tries = 0
        
        while True:
            # m, r = call_model(prompt, model_name=LLM_MODEL)
            r = call_model(prompt, system_message, LLM_MODEL)

            # Check if output is valid JSON
            try:
                responses_json += json.loads(r)
                break
            except json.JSONDecodeError:
                num_tries += 1
                if num_tries >= max_iteration:
                    print(" "*4 + f"Invalid JSON format after {max_iteration} retries. Aborting...")
                    break
                print(" "*4 + "Invalid JSON format. Re-running model...")

    return responses_json


def save_files(responses_json: list, path_json: Path, path_csv: Path):

    print(f"There are {len(responses_json)} pairs generated sentences")
    print(f"Expected number of pairs: {len(issues) * N_PAIRS}")

    print("Saving json file")
    with path_json.open("w") as f:
        json.dump(responses_json, f, indent=4)
    
    print("Saving csv file")
    keys = responses_json[0].keys()
    with path_csv.open("w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(responses_json)


if __name__ == "__main__":

    print("###### Start generating data ######")
    response = generate_data(issues, N_PAIRS)
    print("############## Done ###############")
    
    print("######## Start storing data #######")
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")
    filename = f"synthetic_data_dolphin_{date_string}"
    path_json = Path("./data_generated", filename + ".json")
    path_csv = Path("./data_generated", filename + ".csv")
    save_files(response, path_json, path_csv)
    print("############## Done ###############")
