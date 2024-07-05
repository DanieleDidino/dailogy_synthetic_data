import environ
import sys
from openai import OpenAI
from pathlib import Path

from utils.issues_category import issues
from utils.gen_data_openai import generate_data_openai
from utils.gen_data_ollama import generate_data_ollama
from utils.gen_func_language import convert_functional_language, combine_and_save
from utils.embeddings import get_all_embeddings, create_db, insert_embeddings
from utils.load_save import import_json, save_files

# Import OpenAI key
env = environ.Env()
environ.Env.read_env()
API_KEY = env("OPENAI_API_KEY")

if API_KEY is None:
    print("OpenAI API key is not set. Please set the API_KEY environment variable.")
    sys.exit(1)

# Client
client_openai = OpenAI(api_key=API_KEY)

# Models
LLM_MODEL_OPENAI = "gpt-3.5-turbo" # OpenAI model
TEMPERATURE = 0. # temperature for OpenAI model
LLM_MODEL_OLLAMA = "dolphin-mistral" # Ollama model
EMB_MODEL = "text-embedding-3-small" # Embedding model
N_SENTENCES = 5 # Number of synthetic sentences generate for each issue
FOLDER = "./data_synthetic" # Save here all the files

# Path to synthetic data
filename_openai = f"synthetic_data_{LLM_MODEL_OPENAI}"
path_json_openai = Path(FOLDER, filename_openai + ".json")
path_csv_openai = Path(FOLDER, filename_openai + ".csv")
#
filename_ollama = f"synthetic_data_{LLM_MODEL_OLLAMA}"
path_json_ollama = Path(FOLDER, filename_ollama + ".json")
path_csv_ollama = Path(FOLDER, filename_ollama + ".csv")
#
filename_synthetic_data = "synthetic_data"
path_json_synthetic_data = Path(FOLDER, filename_synthetic_data + ".json")
path_csv_synthetic_data = Path(FOLDER, filename_synthetic_data + ".csv")
#
path_db=Path(FOLDER, "embeddings.db")


def ask_gen_data_gpt():
    prompt_string = f"Generate syntethic data using {LLM_MODEL_OPENAI}? (Y/N)"
    ans = input(prompt_string)
    return ans == "Y"


def ask_gen_data_ollama():
    prompt_string = f"Generate syntethic data using {LLM_MODEL_OLLAMA}? (Y/N)"
    ans = input(prompt_string)
    return ans == "Y"


def ask_functional_text():
    prompt_string = f"Use {LLM_MODEL_OPENAI} to generate functional text? (Y/N)"
    ans = input(prompt_string)
    return ans == "Y"


def ask_emb_sql():
    prompt_string = f"Generate embeddings with {EMB_MODEL} and create SQL database? (Y/N)"
    ans = input(prompt_string)
    return ans == "Y"


if __name__ == "__main__":

    gen_openai = ask_gen_data_gpt()
    if gen_openai:
        print(f"Start generating synthetic data with {LLM_MODEL_OPENAI}")
        response = generate_data_openai(
            issues=issues,
            n_sentences=N_SENTENCES,
            client=client_openai,
            llm_model=LLM_MODEL_OPENAI,
            temperature=TEMPERATURE)
        print(f"Number of sentences generated: {len(issues) * N_SENTENCES}")
        print("Storing data into files...")
        save_files(response, path_json_openai, path_csv_openai)
    
    gen_ollama = ask_gen_data_ollama()
    if gen_ollama:
        print(f"Start generating synthetic data with {LLM_MODEL_OLLAMA}")
        response = generate_data_ollama(
            issues=issues,
            n_sentences=N_SENTENCES,
            llm_model=LLM_MODEL_OLLAMA)
        print(f"Number of sentences generated: {len(issues) * N_SENTENCES}")
        print("Storing data into files...")
        save_files(response, path_json_ollama, path_csv_ollama)
    
    funct_text = ask_functional_text()
    if funct_text:

        print(f"Loading synthetic data denerated with {LLM_MODEL_OPENAI}")
        syn_data_gpt = import_json(path_json_openai)
        print(f"Loading synthetic data denerated with {LLM_MODEL_OLLAMA}")
        syn_data_dolphin = import_json(path_json_ollama)

        print(f"Converting to functional language dataset created with {LLM_MODEL_OPENAI}")
        functional_gpt = convert_functional_language(
            syn_data_gpt,
            client_openai,
            LLM_MODEL_OPENAI,
            TEMPERATURE)
        
        print(f"Converting to functional language dataset created with {LLM_MODEL_OLLAMA}")
        functional_dolphin = convert_functional_language(
            syn_data_dolphin,
            client_openai,
            LLM_MODEL_OPENAI,
            TEMPERATURE)
        
        print(f"Combining {LLM_MODEL_OPENAI} and {LLM_MODEL_OLLAMA} datasets and save into json and csv files")
        combine_and_save(functional_gpt, functional_dolphin, path_json_synthetic_data, path_csv_synthetic_data)
    
    emb_sql = ask_emb_sql()
    if emb_sql:
        file_name_sql=Path("utils/create_bd.sql")
        file_name_bd=Path(FOLDER, "embeddings.db")
        create_db(file_name_sql, file_name_bd)
        print("SQL table created")

        print(f"Loading combined data generated with {LLM_MODEL_OPENAI} and {LLM_MODEL_OLLAMA}")
        synthetic_data = import_json(path_json_synthetic_data)

        print(f"Getting embedding for the synthetic data with {EMB_MODEL}")
        emb_synthetic_data = get_all_embeddings(
            data=synthetic_data,
            model=EMB_MODEL,
            client=client_openai)
        
        print(f"Number of vector embeddings: {len(emb_synthetic_data)}")
        print(f"Length of the vector embedding: {len(emb_synthetic_data[0])}")

        print("Inserting embedding in the sql table")
        insert_embeddings(
            data=synthetic_data,
            embeddings=emb_synthetic_data,
            path_db=path_db)
        
        print("ALL DONE!")
