from pathlib import Path
import os
from pysqlcipher3 import dbapi2 as sqlite
import requests
import zipfile
import io
import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
from concrete.ml.common.serialization.loaders import load

app = Flask(__name__)

#  encrypted database path and encryption key.
DB_PATH = "app_encrypted.db"
ENCRYPTION_KEY = "my_super_secret_key"

# Connect to the encrypted SQLite database and load movies ---
conn = sqlite.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("PRAGMA key = '{}';".format(ENCRYPTION_KEY))
cursor.execute("SELECT title FROM movies ORDER BY id ASC")
rows = cursor.fetchall()
movie_titles = [row[0] for row in rows]
conn.close()

# Load the Compiled FHE Model from the Encrypted Database
conn = sqlite.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("PRAGMA key = '{}';".format(ENCRYPTION_KEY))
cursor.execute("SELECT dump FROM model_dump LIMIT 1")
row = cursor.fetchone()
if row is None:
    raise Exception("No model dump found in the encrypted database.")
model_dump_json = row[0]
conn.close()

# Load the model using Concrete ML's loader.
import io
model_dump_io = io.StringIO(model_dump_json)
fhe_model = load(model_dump_io)

# Recompile the Loaded Model 
if fhe_model.fhe_circuit is None:
    # Selecting a random movie 
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
      { "selection": [0, 1, 0, ..., 0] }  (length 50)
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
    Returns a snippet of the model dump (as stored in the encrypted database)
    to demonstrate that data at rest is encrypted.
    Note: The client sees clear text since the decryption happens on the server.
    """
    conn = sqlite.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA key = '{}';".format(ENCRYPTION_KEY))
    cursor.execute("SELECT dump FROM model_dump LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return jsonify({"error": "Model dump not found"}), 404
    model_dump_json = row[0]
    snippet = model_dump_json[:500]
    return jsonify({"encrypted_model_snippet": snippet})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
