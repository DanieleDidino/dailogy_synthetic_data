import json
import csv
from pathlib import Path


def generate_data_openai(issues: list, n_pairs: int, client, llm_model:str, temperature:float) -> list:
    """
    This function calls the OpenAI API to generate dysfunctional text using as categories the 
    issues listed in the "issues" list. It also generate a functional version of the same text.

    Args:
        issues: A list with the issues to include in the synthetic data query engine.
        n_pairs: Number of synthetic sentences generate for each issue.
        client: A client for the OpenAI API.
        llm_model: Name of the model.
        temperature: Parameter of the OpenAI model.

    Returns:
        responses: A list with dictionaries containing the genrerated dysfunctional text and a tranformed functional version.
    """
    
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
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[
                # {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        responses.append(completion.choices[0].message.content)

    return responses


def save_files_openai(responses: list, path_json: Path, path_csv: Path):
    """
    This function saves the generated synthetic data into a json file and a csv file.

    Args:
        responses: A list with the generated data to store into files.
        path_json: Path for the json file.
        path_csv: Path for the csv file.

    Returns:
        None, save a json file and a csv file.
    """
    responses_json = []

    for response in responses:
        responses_json += json.loads(response)

    print("Saving json file")
    with path_json.open("w") as f:
        json.dump(responses_json, f, indent=4)
    
    print("Saving csv file")
    keys = responses_json[0].keys()
    with path_csv.open("w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(responses_json)
