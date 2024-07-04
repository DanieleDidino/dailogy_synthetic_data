import sqlite3
from pathlib import Path
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# from embeddings import get_embedding
from utils.embeddings import get_embedding


def load_examples(path: Path) -> list[dict]:
    # Fetch embedding from sql database
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, dysfunctional, embedding, functional FROM examples')
    rows = cursor.fetchall()
    conn.close()

    # Move embedding and text into a list
    examples = []
    for row in rows:
        examples.append({
            'id': row[0],
            'dysfunctional': row[1],
            'embedding': np.array(json.loads(row[2])),
            'functional': row[3]
        })

    return examples



def find_closest(input_embedding:list, examples:list[dict], top_n:int=5) -> list[dict]:
    """
    Return top_n pairs of dysfunctional text and its functional version,
    based on the cosine similarity with the input_embedding, which is the
    embedding of the text from the user.

     Args:
        input_embedding: Embedding of the user's text.
        examples: List with dictionaries containing dysfunctional text, the dysfunctinal text embedding, and the functional version.
            It has this structure:
                [
                    {'id': 1,
                    'dysfunctional': "A dysfucntional example",
                    'embedding': array([ 0.03117449,  0.03328631, -0.00667486, ..., -0.01784991,]),
                    'functional': "The functional version of the text"},
                    {'id': 2,
                    'dysfunctional': "A dysfucntional example",
                    'embedding': array([ 0.05065854,  0.01244088, -0.04797346, ...,  0.00821188,]),
                    'functional': "The functional version of the text"},
                    ...
                ]
        top_n: number examples to select.

    Returns:
        selected_examples: A list with dictioraries wiht dysfuntional and functional examples.
            It has this structure:
                [
                    {'dysfunctional': "A dysfucntional example",
                    'functional': "The functional version of the text"},
                    {'dysfunctional': "A dysfucntional example",
                    'functional': "The functional version of the text"},
                    ...
                ]
        selected_similarities: A list with the cosine similarities of the selected dysfunctional examples.
            It is calculated as the cosine similarity between the embeddings of user input and those of the
            dysfunctional examples .
            It has this structure:
                [
                    np.float64(0.6548426546546),
                    np.float64(0.5864792914286),
                    ...
                ]
    """

    example_embeddings = [example['embedding'] for example in examples]
    similarities = cosine_similarity([input_embedding], example_embeddings)[0]
    similar_indices = similarities.argsort()[-top_n:][::-1]

    selected_examples = [{"dysfunctional":examples[i]["dysfunctional"], "functional":examples[i]["functional"]} for i in similar_indices]

    selected_similarities = [similarities[i] for i in similar_indices]

    return selected_examples, selected_similarities


def select_examples(input_text:str, path_emb:Path , emb_model:str, client, num_examples:int=5) -> tuple[list, list]:
    """
    Select the most relevant few-shot examples based on cosine similarity.

    Args:
        data: Dataset with all the text to use to generated the vector embedding.
        path_emb: Path to the .db file with the examples and their embeddings.
        emb_model: Name of the model for the embeddings.
        client: A client for the OpenAI API.
        num_examples: number examples to select.


    Returns:
        dys_text: A list with the dysfuntional examples.
        fun_text: A list with the funtional examples.
    """

    # Embed the user text
    input_embedding = get_embedding(
        text=input_text,
        model=emb_model,
        client=client)
    
    # Load the examples
    examples = load_examples(path_emb)

    # Find the semantically closest example to the input text
    selected_examples, _ = find_closest(input_embedding, examples, num_examples)
   
    return selected_examples


def create_dynamic_prompt(user_text: str, path_emb:Path , emb_model:str, client, num_examples:int=5) -> str:
    """
    Return a prompt based on the user's text and the selected  examples to enter in the prompt as few-shots.

     Args:
        user_text: The user's text.
        path_emb: Path to the .db file with the examples and their embeddings.
        emb_model: Name of the model for the embeddings.
        client: A client for the OpenAI API.
        num_examples: number examples to select.
        

    Returns:
        A string for the dynamic few-shots prompting.
    """

    # Select the examples with higher cosine similarity with the user text.
    # The cosine similarity is calculated between the embedding of the user's text 
    # and the embeddings of the dysfunctional examples.
    # The selected_examples is a list with dictioraries wiht dysfuntional and functional examples,
    # and has this structure:
    #   [
    #       {'dysfunctional': "A dysfucntional example",
    #       'functional': "The functional version of the text"},
    #       {'dysfunctional': "A dysfucntional example",
    #       'functional': "The functional version of the text"},
    #       ...
    #   ]
    selected_examples = select_examples(
        input_text=user_text,
        path_emb=path_emb,
        emb_model=emb_model,
        client=client,
        num_examples=num_examples)

    prompt_1 = """
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
    
    ### Examples 
    Here are some examples of how to convert a dysfucntional text into functional version:
    """

    prompt_2 = ""

    for example in selected_examples:
        example_text = f"""
        - Input: {example["dysfunctional"]}
        - Expected Output: {example["functional"]}
        """
        prompt_2 += example_text


    prompt_3 = f"""
    ### Input 
    Please transform the following text into functional language:
    
    {user_text}
    """

    prompt = prompt_1 + prompt_2 + prompt_3

    return prompt
