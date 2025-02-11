FROM python:3.8-slim

# Install system dependencies required by pysqlcipher3 (SQLCipher development libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlcipher-dev \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements.txt first for caching.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# Run the training script to generate model
RUN python train_model.py

# Expose the Flask port.
EXPOSE 5000

# Run the Flask application with Waitress.
CMD ["waitress-serve", "--port=5000", "app:app"]
