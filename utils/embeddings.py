import json
import sys
from pathlib import Path
import sqlite3


def import_json(file_path:Path):
    print(f"Loading file: {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_embedding(text: str, model:str, client) -> list:
    """
    Generate embeddings for the input text using OpenAI's API.

    Args:
        text: Text to use to generated the vector embedding.
        model: Name of the model for the embeddings.
        client: A client for the OpenAI API.

    Returns:
        A list with the vector embedding.
    """
    response = client.embeddings.create(input = [text], model=model)
    return response.data[0].embedding


def get_all_embeddings(data:list, model:str, client):
    """
    Generate embeddings for all the text in 'data'.
    'data' is a list of dictionaries, for example:
        [
            {'dysfunctional': "Text to embed",
            'functional': "Functional version of the Text to embed, we do not embed this text"},
            {'dysfunctional': "Next text to embed",
            'functional': "Functional version, we do not embed this text"}
        ]

    Args:
        data: Dataset with all the text to use to generated the vector embedding.
        model: Name of the model.
        client: A client for the OpenAI API.

    Returns:
        A list with lists of vector embeddings for different text in data.
    """
    return [get_embedding(text=text["dysfunctional"], model=model, client=client) for text in data]


def create_db(file_name_sql:str, file_name_bd:str):
    """
    Create a SQL database to store the embeddings

    Args:
        file_name_sql: Name of the file with the SQL script.
        file_name_bd: Name of the file for the database.

    Returns:
        Create a .db file for inserting the vector embeddings.
    """

    file_path = Path(file_name_bd)

    if file_path.exists():
        print("The table already exists! Embeddings not inserted!")
        sys.exit(0)
    else:
        print("Creating table...")
        with open(file_name_sql, 'r') as sql_file:
            sql_script = sql_file.read()
        db = sqlite3.connect(file_name_bd)
        cursor = db.cursor()
        cursor.executescript(sql_script)
        db.commit()
        db.close()


def insert_embeddings(data:list, embeddings:list, path_db:str):
    """
    This function insert the embeddings in a .db file.
    'data' is a list of dictionaries, for example:
        [
            {'dysfunctional': "Text to embed",
            'functional': "Functional version of the Text to embed, we do not embed this text"},
            {'dysfunctional': "Next text to embed",
            'functional': "Functional version, we do not embed this text"}
        ]

    Args:
        data: List with the dysfunctional text (used to generated the vector embedding) 
              and the functional version.
        embeddings: List with the embedding for the dysfunctional text.
        path_db: Path to the .db file in which insert the text and embeddings.

    Returns:
        Create a .db file for inserting the vector embeddings.
    """
    con = sqlite3.connect(path_db)
    with con:
        for ex, emb in zip(data, embeddings):
            con.execute(
                "INSERT INTO examples (dysfunctional, embedding, functional) VALUES (?, ?, ?)",
                (ex["dysfunctional"], json.dumps(emb), ex["functional"])
            )
    con.close()
