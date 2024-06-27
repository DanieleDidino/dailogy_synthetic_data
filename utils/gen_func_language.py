import json
import csv
from pathlib import Path

def create_prompt(text:str) -> str:
    prompt = f"""
    Below is an instruction that describes a task.
    Write a response that appropriately completes the request.
    
    ### Objective:
    Transform the following text, which originates from the context of dysfunctional communication between couples, into functional language.
    Make the text actionable or practical, while maintaining a natural, conversational tone.
    
    ### Instructions:
    1. Review the provided text carefully.
    2. Convert the text into functional, everyday language, focusing on making the content actionable and practical.
    3. Aim for a conversational tone, as if explaining to a friend, to ensure the paragraph is engaging and accessible.
    4. Ensure the transformed text promotes understanding, empathy, and positive communication, suitable for couples or ex-couples who need to interact constructively.
    5. Always respond only with the transformed text and nothing else.
    
    ### Input 
    Please transform the following text into functional language:
    
    {text["dysfunctional"]}
    """
    return prompt


def call_client(prompt:str, client_openai, llm_model:str, temperature:float) -> str:
    completion = client_openai.chat.completions.create(
        model=llm_model,
        messages=[
            # {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return completion.choices[0].message.content


def pair_text(data_input:list[dict], responses:list) -> list:
    paired_text = []

    for dysfunctional, functional in zip(data_input, responses):
        paired_text.append({
            'dysfunctional': dysfunctional["dysfunctional"],
            'functional': functional
        })

    return paired_text


def convert_functional_language(data:list[dict], client_openai, llm_model:str, temperature: float) -> list:

    responses = []

    for text in data:

        prompt = create_prompt(text)
        response = call_client(prompt, client_openai, llm_model, temperature)
        responses.append(response)

    print(f"Length input: {len(data)}")
    print(f"Length output: {len(responses)}")

    return pair_text(data, responses)


def combine_and_save(data1:list[dict], data2:list[dict], path_json: Path, path_csv: Path):
    """
    Combine 2 objects in json fomat into a json file and a csv file.
    """

    data_combined = data1 + data2

    print("Saving combined data into a json file")
    with path_json.open("w") as f:
        json.dump(data_combined, f, indent=4)
    
    print("Saving combined data into a csv file")
    keys = data_combined[0].keys()
    with path_csv.open("w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_combined)

    return json.dumps(data1 + data2, indent=4)
