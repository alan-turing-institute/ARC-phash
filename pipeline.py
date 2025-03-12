import os
import sqlite3

import numpy as np
from PIL import Image


def load_image(image_path):
    return Image.open(image_path)


def embed_image(image):
    """Dummy embedding function.

    Resize the image, flatten, and normalize pixel values.
    Replace this with actual embedding model later.
    """
    image = image.resize((64, 64))
    arr = np.array(image).flatten().astype(np.float32)
    norm = np.linalg.norm(arr) or 1.0
    return (arr / norm).tolist()


def setup_db(db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    return conn


def save_embedding(conn, image_name, embedding):
    c = conn.cursor()
    # Store embedding as a string for prototype simplicity
    c.execute(
        "INSERT INTO embeddings (image_name, embedding) VALUES (?, ?)",
        (image_name, str(embedding)),
    )
    conn.commit()


def main():
    data_folder = "data/images"
    conn = setup_db()

    for image_file in os.listdir(data_folder):
        image_path = os.path.join(data_folder, image_file)
        try:
            image = load_image(image_path)
            embedding = embed_image(image)
            save_embedding(conn, image_file, embedding)
            print(f"Processed {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    main()
