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


def generate_data_ollama(issues:list, n_sentences:int, llm_model:str, max_iteration:int=5) -> list:
    """
    This function uses the Ollama framework to generate dysfunctional text using as categories the 
    issues listed in the "issues" list.
    The model output from the Ollama model was unstable and frequently did not conform to a JSON file format.
    
    To resolve this, I implemented the following:
    - Introduced a 'system message'.
    - Reiterated the desired format in the 'prompt'.
    
    But also with these adjustments, the model now does not generate the desired output format.
    So I added a for loop with 'max_iteration' iterations, that is a way to check if the output
    is correct and if not it re-run the model.

    Args:
        issues: A list with the issues to include in the synthetic data query engine.
        n_sentences: Number of synthetic sentences generate for each issue.
        llm_model: Name of the model.
        max_iteration: Max number of iteration to try before aborting the function.

    Returns:
        responses: A list with a dictionaries containing the generated dysfunctional text.

    """

    # List to store the responses
    responses = []

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
    
        Each entry should generate a sentence reflecting dysfunctional communication, showcasing various forms
        of toxicity such as insults, harassment, threats, manipulation, and derogatory remarks.
    
        Ensure the sentences are realistic and diverse in terms of content and context.
        The sentences should refer to this issue category:
        '{issue}'
    
        Provide {n_sentences} sentences.
        """
    
        # The prompt is divided into 2 sub-prompts because
        # the example of the output format uses Curly brackets {},
        # and this cannot be done in a f-string.
        prompt_2 = """
        Always respond only with valid JSON format and nothing else.
        Do not include any text before or after the JSON.

        You must provide the output exactly in the following format:
        
        [
            {"dysfunctional": "write here the dysfunctional text"}
            {"dysfunctional": "write here the dysfunctional text"},
        ]
        """
    
        prompt = prompt_1 + prompt_2

        num_tries = 0

        while True:
            r = call_model(prompt, system_message, llm_model)
        
            # Check if output is valid JSON
            try:
                responses += json.loads(r)
                break
            except json.JSONDecodeError:
                num_tries += 1
                if num_tries >= max_iteration:
                    print(" "*4 + f"Invalid JSON format after {max_iteration} retries. Aborting...")
                    break
                print(" "*4 + "Invalid JSON format. Re-running model...")

    return responses
