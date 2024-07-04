import environ
import sys
from openai import OpenAI
from pathlib import Path
from datetime import datetime

from utils.dynamic_prompt import create_dynamic_prompt

# Import OpenAI key
env = environ.Env()
environ.Env.read_env()
API_KEY = env("OPENAI_API_KEY")

if API_KEY is None:
    print("OpenAI API key is not set. Please set the API_KEY environment variable.")
    sys.exit(1)

# Client
client_openai = OpenAI(api_key=API_KEY)

# Embedding model
EMB_MODEL = "text-embedding-3-small" # Embedding model

# Path to embedding database
FOLDER = "./data_synthetic" # folder wiht generated synthetic data
PATH_EMB_DB = Path(FOLDER, "embeddings.db")
# Number of example to use as few-shots in the prompt
NUM_EXAMPLES_TO_SELECT = 5

# Folder to save the prompts
PATH_PROMPTS = "dynamic_fewshot_prompts"


def ask_input():
    prompt_string = "You can input a new text or use a defaul example. Input text? (Y/N)"
    ans = input(prompt_string)
    return ans == "Y"


if __name__ == "__main__":

    input_text = ask_input()
    if input_text:
        prompt_string = "Write the text to add to the prompt:"
        text = input(prompt_string)
    else:
        text = "Your poor decisions regarding our child's health show your laziness, putting all the responsibility on me."
    
    print(f"This text will be added to the prompt:\n{text}")

    print("Creating dynamic few-shot prompt...")
    dynamic_fewshot_prompt = create_dynamic_prompt(
        user_text=text,
        path_emb=PATH_EMB_DB,
        emb_model=EMB_MODEL,
        client=client_openai,
        num_examples=NUM_EXAMPLES_TO_SELECT)
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    filename = f"prompt_{current_time}.txt"
    path_prompt = Path(PATH_PROMPTS, filename)
    
    print(f"Saving the prompt into a file...")
    with open(path_prompt, "w") as file:
        file.write(dynamic_fewshot_prompt)
    
    print(f"File saved as {str(path_prompt)}")

    print("ALL DONE!")
