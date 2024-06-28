import json
import csv
from pathlib import Path


def generate_data_openai(issues: list, n_pairs: int, client, llm_model:str, temperature:float, max_iteration:int=5) -> list:
    """
    This function calls the OpenAI API to generate dysfunctional text using as categories the 
    issues listed in the "issues" list.

    The model output sometimes did not conform to the JSON file format.
    To resolve this, I added a for loop with 'max_iteration' iterations.
    This loop checks if the output is correct, and if not, it re-runs the model.

    Args:
        issues: A list with the issues to include in the synthetic data query engine.
        n_pairs: Number of synthetic sentences generate for each issue.
        client: A client for the OpenAI API.
        llm_model: Name of the model.
        temperature: Parameter of the OpenAI model.
        max_iteration: Max number of iteration to try before aborting the function.

    Returns:
        responses: A list with dictionaries containing the genrerated dysfunctional text.
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

        Each entry should generate a sentence reflecting dysfunctional communication, showcasing various forms
        of toxicity such as insults, harassment, threats, manipulation, and derogatory remarks.
    
        Ensure the sentences are realistic and diverse in terms of content and context.
        The sentences should refer to this issue category:
        '{issue}'
    
        Provide {n_pairs} sentences.
        """
    
        # The prompt is divided into 2 sub-prompts because
        # the example of the output format uses Curly brackets {},
        # and this cannot be done in a f-string.
        prompt_2 = """
        Write the output using this format:
        [
            {"dysfunctional": "write here the dysfunctional text"},
            {"dysfunctional": "write here the dysfunctional text"},
        ]
        """
    
        prompt = prompt_1 + prompt_2

        num_tries = 0
        
        while True:
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[
                    # {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            r = completion.choices[0].message.content
                    
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
