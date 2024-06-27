import environ
import sys
from openai import OpenAI
from pathlib import Path

from utils.issues_category import issues
from utils.gen_data_openai import generate_data_openai, save_files_openai
from utils.gen_data_ollama import generate_data_ollama, save_files_ollama
from utils.gen_func_language import convert_functional_language, combine_and_save
from utils.embeddings import get_all_embeddings, create_db, insert_embeddings
from utils.load_save import import_json

# Import OpenAI key
env = environ.Env()
environ.Env.read_env()
API_KEY = env("OPENAI_API_KEY")

if API_KEY is None:
    print("OpenAI API key is not set. Please set the API_KEY environment variable.")
    sys.exit(1)

# Client
client_openai = OpenAI(api_key=API_KEY)

LLM_MODEL_OPENAI = "gpt-3.5-turbo" # OpenAI model
TEMPERATURE = 0. # temperature for OpenAI model
LLM_MODEL_OLLAMA = "dolphin-mistral" # Ollama model
EMB_MODEL = "text-embedding-3-small" # Embedding model
N_PAIRS = 5 # NUmber of synthetic sentences generate for each issue
FOLDER = "./data_synthetic" # Save here all the files

# Path to save synthetic data
synthetic_data_file_name = "synthetic_data"
synthetic_data_path_json = Path(FOLDER, synthetic_data_file_name + ".json")
synthetic_data_path_csv = Path(FOLDER, synthetic_data_file_name + ".csv")


def ask_gen_data_gpt():
    prompt_string = "Generate syntethic data using a OpenAI API? (Y/N)"
    ans = input(prompt_string)
    return ans == "Y"


def ask_gen_data_ollama():
    prompt_string = "Generate syntethic data using Ollama? (Y/N)"
    ans = input(prompt_string)
    return ans == "Y"


def ask_functional_text():
    prompt_string = "Use OpenAI client to generate functional text? (Y/N)"
    ans = input(prompt_string)
    return ans == "Y"


def ask_emb_sql():
    prompt_string = "Generate embeddings and create SQL database? (Y/N)"
    ans = input(prompt_string)
    return ans == "Y"


if __name__ == "__main__":

    gen_openai = ask_gen_data_gpt()
    if gen_openai:
        print("Start generating synthetic data with OpenAI API")
        response = generate_data_openai(
            issues=issues,
            n_pairs=N_PAIRS,
            client=client_openai,
            llm_model=LLM_MODEL_OPENAI,
            temperature=TEMPERATURE)
        print(f"Number of sentences generated: {len(issues) * N_PAIRS}")
        print("Storing data into files...")
        filename = "synthetic_data_gpt"
        path_json = Path(FOLDER, filename + ".json")
        path_csv = Path(FOLDER, filename + ".csv")
        save_files_openai(response, path_json, path_csv)
    
    gen_ollama = ask_gen_data_ollama()
    if gen_ollama:
        print("Start generating synthetic data with Ollama")
        response = generate_data_ollama(
            issues=issues,
            n_pairs=N_PAIRS,
            llm_model=LLM_MODEL_OLLAMA)
        print(f"Number of sentences generated: {len(issues) * N_PAIRS}")
        print("Storing data into files...")
        filename = f"synthetic_data_dolphin"
        path_json = Path(FOLDER, filename + ".json")
        path_csv = Path(FOLDER, filename + ".csv")
        save_files_ollama(response, path_json, path_csv)
    
    # Folder and files with synthetic data
    file_name_gpt = "synthetic_data_gpt.json"
    file_path_gpt = Path(FOLDER, file_name_gpt)
    file_name_dolphin = "synthetic_data_dolphin.json"
    file_path_dolphin = Path(FOLDER, file_name_dolphin)
    
    funct_text = ask_functional_text()
    if funct_text:

        print("Loading synthetic data denerated with gpt")
        syn_data_gpt = import_json(file_path_gpt)
        print("Loading synthetic data denerated with dolphin")
        syn_data_dolphin = import_json(file_path_dolphin)

        print("Converting to functional language dataset created with gpt")
        functional_gpt = convert_functional_language(
            syn_data_gpt,
            client_openai,
            LLM_MODEL_OPENAI,
            TEMPERATURE)
        
        print("Converting to functional language dataset created with dolphin")
        functional_dolphin = convert_functional_language(
            syn_data_dolphin,
            client_openai,
            LLM_MODEL_OPENAI,
            TEMPERATURE)
        
        print("Combining gpt and dolphin datasets and save into json and csv files")
        combine_and_save(functional_gpt, functional_dolphin, synthetic_data_path_json, synthetic_data_path_csv)
    
    emb_sql = ask_emb_sql()
    if emb_sql:
        file_name_sql=Path("utils/create_bd.sql")
        file_name_bd=Path(FOLDER, "embeddings.db")
        create_db(file_name_sql, file_name_bd)
        print("SQL table created")

        print("Loading combined data generated with gpt and dolphin")
        synthetic_data = import_json(synthetic_data_path_json)

        print("Getting embedding for the synthetic data")
        emb_synthetic_data = get_all_embeddings(
            data=synthetic_data,
            model=EMB_MODEL,
            client=client_openai)
        
        print(f"Loaded {len(emb_synthetic_data)} sentences generated with gpt")
        print(f"Length of the vector embeddings: {len(emb_synthetic_data[0])}")

        path_db=Path(FOLDER, "embeddings.db")
        print("Inserting embedding for gpt example in the sql table")
        insert_embeddings(
            data=synthetic_data,
            embeddings=emb_synthetic_data,
            path_db=path_db)
