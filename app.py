from pathlib import Path
import os
import requests
import zipfile
import io
import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
from concrete.ml.common.serialization.loaders import load

app = Flask(__name__)

# --- Ensure movies.csv exists; download if necessary ---
if not os.path.exists("movies.csv"):
    app.logger.info("movies.csv not found. Downloading MovieLens dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract movies.csv (located at ml-latest-small/movies.csv)
            z.extract("ml-latest-small/movies.csv", path=".")
            os.rename("ml-latest-small/movies.csv", "movies.csv")
        app.logger.info("Downloaded and extracted movies.csv.")
    else:
        raise Exception("Failed to download MovieLens dataset.")

# --- Load the first 50 movies ---
movies_df = pd.read_csv("movies.csv")
movies_df = movies_df.head(50)
movie_titles = movies_df["title"].tolist()

# --- Load the Compiled FHE Model from JSON ---
# This file is produced by your training script (using model.dump).
dumped_model_path = Path("logistic_regression_model.json")
with dumped_model_path.open("r") as f:
    fhe_model = load(f)

# --- Recompile the Loaded Model if Needed ---
if fhe_model.fhe_circuit is None:
    # Generate calibration data with shape (1, 50); in practice use representative calibration data.
    calibration_data = np.random.randint(0, 2, size=(1, 50)).astype(np.float64)
    fhe_model.compile(calibration_data)
    app.logger.info("Recompiled the loaded model with calibration data.")

@app.route("/")
def index():
    return render_template("index.html", movies=movie_titles)

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Expects a JSON payload:
      { "selection": [0, 1, 0, 1, ..., 0] }
    where "selection" is a binary vector of length 50.
    Uses the model's predict method with fhe="execute" to perform FHE inference.
    Returns the recommended movie index and title.
    """
    try:
        data = request.get_json()
        selection = data.get("selection")
        selection = [int(x) for x in selection]
        if len(selection) != 50:
            raise ValueError("Selection vector must be of length 50.")
    except Exception as e:
        app.logger.exception("Invalid input:")
        return jsonify({"error": "Invalid input", "details": str(e)}), 400

    try:
        # The predict method expects a 2D array.
        prediction = fhe_model.predict([selection], fhe="execute")
    except Exception as e:
        app.logger.exception("Inference error:")
        return jsonify({"error": "Inference error", "details": str(e)}), 500

    try:
        recommended_index = int(prediction[0])
    except Exception as e:
        app.logger.exception("Prediction processing error:")
        return jsonify({"error": "Prediction processing error", "details": str(e)}), 500

    if 0 <= recommended_index < len(movie_titles):
        recommended_movie = movie_titles[recommended_index]
    else:
        recommended_movie = "Unknown"
    return jsonify({"prediction": recommended_index, "recommendation": recommended_movie})

@app.route("/demo", methods=["GET"])
def demo():
    """
    Returns a snippet of the dumped model file to demonstrate that the model data is stored
    in an obfuscated/encrypted format.
    """
    dumped_model_path = Path("logistic_regression_model.json")
    if dumped_model_path.exists():
        with dumped_model_path.open("r") as f:
            content = f.read()
        snippet = content[:500]  # Return the first 500 characters
        return jsonify({"encrypted_model_snippet": snippet})
    else:
        return jsonify({"error": "Model dump file not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
