import json
import csv
from pathlib import Path
import ollama


def call_model(prompt:str, system_message:str, model_name:str) -> list:
    """
    Take a prompt and a model name and use the ollama framework to generate a response.

    Args:
        prompt: Prompt to provide to the model.
        system_message: System message to provide to the model.
        model_name: Name of the model.

    Returns:
        responses: A list with dictionaries containing the genrerated dysfunctional text and a tranformed functional version.
    """

    response = ollama.generate(
            model=model_name,
            prompt=prompt,
            system=system_message)
        
    return response["response"]


def generate_data_ollama(issues:list, n_pairs:int, llm_model:str, max_iteration:int=5) -> list:
    """
    This function uses the Ollama framework to generate dysfunctional text using as categories the 
    issues listed in the "issues" list. It also generate a functional version of the same text.
    The model output from the Ollama model was unstable and frequently did not conform to a JSON file format.
    
    To resolve this, I implemented the following:
    - Introduced a 'system message'.
    - Reiterated the desired format in the 'prompt'.
    
    But also with these adjustments, the model now does not generate the desired output format.
    So I added a for loop with 'max_iteration' iterations, that is a way to check if the output
    is correct and if not it re-run the model.

    Args:
        issues: A list with the issues to include in the synthetic data query engine.
        n_pairs: Number of synthetic sentences generate for each issue.
        llm_model: Name of the model.
        max_iteration: Max number of iteration to try before aborting the function.

    Returns:
        responses: A list with dictionaries containing the genrerated dysfunctional text and a tranformed functional version.

    """

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

        num_tries = 0

        while True:
            r = call_model(prompt, system_message, llm_model)

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


def save_files_ollama(responses:list, path_json:Path, path_csv:Path):
    """
    This function saves the generated synthetic data into a json file and a csv file.

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
