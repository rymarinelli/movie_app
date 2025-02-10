# Dockerfile
FROM python:3.8-slim

# Install system dependencies required by Concrete ML.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /app

# Copy requirements and install Python packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including code, templates, etc.)
COPY . .

# Run the training script so that fhe_model_concrete.pkl is generated.
RUN python train_model.py

# Expose the Flask port.
EXPOSE 5000

# Run the Flask application.
CMD ["python", "app.py"]
