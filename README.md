# 🎜 Movie App




A simple **Flask web application** that serves movie-related content.\
The repo also includes deployment files for **Docker** and **Kubernetes**.

---

## 📸 Screenshot

\
*(replace with your own screenshot)*

---

## 🚀 Getting Started

### Local (Python)

**Requirements:** Python 3.9+ and Git.

```bash
# Clone the repo
git clone https://github.com/rymarinelli/movie_app
cd movie_app

# Create and activate a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the app
python app.py
```

Open: [http://localhost:5000](http://localhost:5000)

---

### Docker

**Requirements:** Docker Desktop or Docker Engine.

```bash
# Build the image
docker build -t movie-app:latest .

# Run the container
docker run --rm -p 5000:5000 movie-app:latest
```

Open: [http://localhost:5000](http://localhost:5000)

---

### Kubernetes

**Requirements:** A Kubernetes cluster and `kubectl`.\
For local testing, [minikube](https://minikube.sigs.k8s.io/docs/) is easiest.

#### Using Minikube

```bash
# Start minikube
minikube start

# Use minikube's Docker daemon so you don’t need a registry
eval $(minikube docker-env) # Windows PowerShell: & minikube -p minikube docker-env | Invoke-Expression

# Build the image
docker build -t movie-app:latest .

# Apply manifests
kubectl apply -f movie-app-deployment.yaml
kubectl apply -f movie-app-service.yaml

# Open service in browser
minikube service movie-app-service
```

#### Using a Remote Cluster

1. Push the image to a registry (e.g., Docker Hub, GHCR):

```bash
docker tag movie-app:latest YOUR_REGISTRY/movie-app:latest
docker push YOUR_REGISTRY/movie-app:latest
```

2. Update `movie-app-deployment.yaml` to use the pushed image.
3. Deploy:

```bash
kubectl apply -f movie-app-deployment.yaml
kubectl apply -f movie-app-service.yaml
kubectl get svc movie-app-service
```

Access via the external IP/port shown.

---

## 📂 Project Structure

```
movie_app/
├── app.py                   # Flask app entry point
├── templates/               # HTML templates
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker build instructions
├── movie-app-deployment.yaml# Kubernetes Deployment
├── movie-app-service.yaml   # Kubernetes Service
└── (other scripts)          # Experimental / optional
```

---

## ⚠️ Common Issues

- **Port already in use** → Change host port (`-p 8080:5000`).
- **Module not found** → Ensure your virtual environment is activated and `pip install -r requirements.txt` ran.
- **ImagePullBackOff in Kubernetes** → Push the image to a registry accessible to your cluster, or use minikube’s Docker daemon.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
