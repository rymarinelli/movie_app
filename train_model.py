import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path
from pysqlcipher3 import dbapi2 as sqlite # Use pysqlcipher3 to work with an encrypted SQLite database.
from concrete.ml.sklearn import LogisticRegression as FHELogisticRegression

# Define the database file and encryption key.
DB_PATH = "app_encrypted.db"
ENCRYPTION_KEY = "my_super_secret_key" 

# Connect to (or create) the encrypted SQLite database.
conn = sqlite.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("PRAGMA key = '{}';".format(ENCRYPTION_KEY))
cursor.execute("PRAGMA cipher_compatibility = 4;")
conn.commit()

# Create the movies table if it doesn't exist.
cursor.execute('''
    CREATE TABLE IF NOT EXISTS movies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL
    )
''')
conn.commit()

# Create the model_dump table. Mostly to be used by \demo endpoint at presentations
cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_dump (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dump TEXT NOT NULL
    )
''')
conn.commit()


if not os.path.exists("movies.csv"):
    print("movies.csv not found. Downloading MovieLens dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extract("ml-latest-small/movies.csv", path=".")
            os.rename("ml-latest-small/movies.csv", "movies.csv")
        print("Downloaded and extracted movies.csv.")
    else:
        raise Exception("Failed to download MovieLens dataset.")


movies_df = pd.read_csv("movies.csv")
movies_df = movies_df.head(50)
movie_titles = movies_df["title"].tolist()
print("Loaded movie catalog (50 movies):")
print(movie_titles)


cursor.execute("SELECT COUNT(*) FROM movies")
if cursor.fetchone()[0] == 0:
    for title in movie_titles:
        cursor.execute("INSERT INTO movies (title) VALUES (?)", (title,))
    conn.commit()
    print("Inserted 50 movies into the encrypted database.")
else:
    print("Movies already exist in the encrypted database.")

n_movies = 50
n_samples = 1000  


#Example Training Process
X = np.random.randint(0, 2, size=(n_samples, n_movies)).astype(np.float64)
y = []
for row in X:
    not_selected = [i for i, val in enumerate(row) if val == 0]
    if not not_selected:
        recommended = np.random.randint(0, n_movies)
    else:
        recommended = np.random.choice(not_selected)
    y.append(recommended)
y = np.array(y)


model = FHELogisticRegression(n_bits=8, multi_class='ovr', max_iter=1000)
model.fit(X, y)

# Compile the Model for FHE Inference ---
calibration_data = np.random.randint(0, 2, size=(1, n_movies)).astype(np.float64)
model.compile(calibration_data)

# Dump model to a JSON
import io
model_dump_io = io.StringIO()
model.dump(model_dump_io)
model_dump_json = model_dump_io.getvalue()

# Remove any existing model dump and insert the new dump into the database.
cursor.execute("DELETE FROM model_dump")
cursor.execute("INSERT INTO model_dump (dump) VALUES (?)", (model_dump_json,))
conn.commit()

print("Model trained, compiled, and saved to the encrypted database (model_dump table).")
conn.close()
