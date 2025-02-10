import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path
from concrete.ml.sklearn import LogisticRegression as FHELogisticRegression

# --- Download MovieLens dataset if movies.csv does not exist ---
if not os.path.exists("movies.csv"):
    print("movies.csv not found. Downloading MovieLens dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract the movies.csv file (located at ml-latest-small/movies.csv)
            z.extract("ml-latest-small/movies.csv", path=".")
            os.rename("ml-latest-small/movies.csv", "movies.csv")
        print("Downloaded and extracted movies.csv.")
    else:
        raise Exception("Failed to download MovieLens dataset.")

# --- Load the first 50 movies from movies.csv ---
movies_df = pd.read_csv("movies.csv")
movies_df = movies_df.head(50)
movie_titles = movies_df["title"].tolist()
print("Loaded movie catalog (50 movies):")
print(movie_titles)

n_movies = len(movie_titles)  # Should be 50
n_samples = 1000              # Number of synthetic training samples

# --- Generate Synthetic Training Data ---
# Each training sample is a binary vector of length n_movies (50 in this case).
X = np.random.randint(0, 2, size=(n_samples, n_movies))
# Convert X to float (required for quantization)
X = X.astype(np.float64)

y = []
for row in X:
    # Choose a movie (by index) that was NOT selected.
    not_selected = [i for i, val in enumerate(row) if val == 0]
    if not not_selected:
        # If the user selected all movies, choose one at random.
        recommended = np.random.randint(0, n_movies)
    else:
        recommended = np.random.choice(not_selected)
    y.append(recommended)
y = np.array(y)

# --- Train a Multi-Class Logistic Regression Model ---
model = FHELogisticRegression(n_bits=8, multi_class='ovr', max_iter=1000)
model.fit(X, y)

# --- Compile the Model for FHE Inference ---
# Use calibration data with the correct shape (1, 50)
calibration_data = np.random.randint(0, 2, size=(1, n_movies)).astype(np.float64)
model.compile(calibration_data)

# --- Dump the Compiled Model to a JSON File ---
# (Assuming your Concrete ML version supports dumping the model to JSON.)
dumped_model_path = Path("logistic_regression_model.json")
with dumped_model_path.open("w") as f:
    model.dump(f)

print("Model trained, compiled, and dumped to logistic_regression_model.json")
