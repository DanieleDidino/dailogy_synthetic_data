import environ
import json
from pathlib import Path
from openai import OpenAI
import sqlite3

# Import OpenAI key
env = environ.Env()
environ.Env.read_env()
API_KEY = env("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Embedding model
EMB_MODEL = "text-embedding-3-small" # emb_model_name


def import_json(file_path:Path):
    # Select one of the files
    print(f"Loading file: {file_path}")
    with open(file_path, 'r') as file:
        syn_data = json.load(file)
    return syn_data


def get_embedding(text: str, model:str):
    """
    Generate embeddings for the input text using OpenAI's API.
    """
    response = client.embeddings.create(input = [text], model=model)
    return response.data[0].embedding


def get_all_embeddings(data:list, model:str):
    return [get_embedding(text=text["dysfunctional"], model=model) for text in data]


def ask_sql_bd():
    prompt_string = "Create sql database? (Y/N)"
    write_bd = input(prompt_string)
    return write_bd == "Y"


def create_db(file_name_sql:str, file_name_bd:str):

    file_path = Path(file_name_bd)

    if file_path.exists():
        print("The table already exists!")
        return None
    else:
        print("Creating table...")
        with open(file_name_sql, 'r') as sql_file:
            sql_script = sql_file.read()
        db = sqlite3.connect(file_name_bd)
        cursor = db.cursor()
        cursor.executescript(sql_script)
        db.commit()
        db.close()


def ask_insert_emb():
    prompt_string = "Insert embeddings in the sql database? (Y/N)"
    insert_emb = input(prompt_string)
    return insert_emb == "Y"


def insert_embeddings(examples:list, embeddings:list, path_db:str):
    con = sqlite3.connect(path_db)
    with con:
        for ex, emb in zip(examples, embeddings):
            con.execute(
                "INSERT INTO examples (dysfunctional, embedding, functional) VALUES (?, ?, ?)",
                (ex["dysfunctional"], json.dumps(emb), ex["functional"])
            )
    con.close()


if __name__ == "__main__":

    # Folder and files with synthetic data
    folder_db = "data_generated/"
    file_name_gpt = "synthetic_data_gpt_2024-06-18_11-27.json"
    file_path_gpt = Path(folder_db, file_name_gpt)
    file_name_dolphin = "synthetic_data_dolphin_2024-06-18_11-46.json"
    file_path_dolphin = Path(folder_db, file_name_dolphin)

    print("Loading synthetic data denerated with gpt")
    syn_data_gpt = import_json(file_path_gpt)
    print("Loading synthetic data denerated with dolphin")
    syn_data_dolphin = import_json(file_path_dolphin)

    print("Getting embedding for the synthetic data denerated with gpt")
    emb_gpt = get_all_embeddings(data=syn_data_gpt, model=EMB_MODEL)
    print("Getting embedding for the synthetic data denerated with dolphin")
    emb_dolphin = get_all_embeddings(data=syn_data_dolphin, model=EMB_MODEL)

    print(f"Loaded {len(emb_gpt)} sentences generated with gpt")
    print(f"Loaded {len(emb_dolphin)} sentences generated with dolphin")
    print(f"Length of the vector embeddings: {len(emb_gpt[0])}")

    write_db = ask_sql_bd()
    if write_db:
        file_name_sql=Path("utils/create_bd.sql")
        file_name_bd=Path("database/emb_examples.db")
        create_db(file_name_sql, file_name_bd)
        print("SQL table created")
    else:
        print("SQL table not created")
    
    insert_emb = ask_insert_emb()
    if insert_emb:
        path_db=Path("database/emb_examples.db")
        print("Inserting embedding for gpt example in the sql table")
        insert_embeddings(examples=syn_data_gpt, embeddings=emb_gpt, path_db=path_db)
        print("Inserting embedding for dolhin example in the sql table")
        insert_embeddings(examples=syn_data_dolphin, embeddings=emb_dolphin, path_db=path_db)
    else:
        print("Embeddings not inserted in the sql table")
