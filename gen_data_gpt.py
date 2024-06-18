import environ
import sys
from openai import OpenAI
import openai
import json
import csv
from pathlib import Path
from datetime import datetime

from synthetic_data.issues_category import issues

# Import OpenAI key
env = environ.Env()
environ.Env.read_env()
API_KEY = env("OPENAI_API_KEY")
openai.api_key = API_KEY

# Client
client_openai = OpenAI()

# Set constanst
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0

# How many synthetic sentences generate for each issue
N_PAIRS = 5


def generate_data(issues: list, n_pairs: int) -> list:

    if API_KEY is None:
        print("OpenAI API key is not set. Please set the API_KEY environment variable.")
        sys.exit(1)
    
    # List to store the reposnes
    responses = []

    # Count the iteration throught the "issues" list
    N_issue = 1

    for issue in issues:

        # Print the current issue number
        print(f"Generating output {N_issue} of {len(issues)}")
        N_issue += 1

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
        Write the output using this format:
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
        
        print(" "*4 + "Running model...")
        completion = client_openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                # {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
        )
        responses.append(completion.choices[0].message.content)

    return responses


def save_files(responses: list, path_json: Path, path_csv: Path):
    responses_json = []

    for response in responses:
        responses_json += json.loads(response)

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
    filename = f"synthetic_data_gpt_{date_string}"
    path_json = Path("./synthetic_data", filename + ".json")
    path_csv = Path("./synthetic_data", filename + ".csv")
    save_files(response, path_json, path_csv)
    print("############## Done ###############")
