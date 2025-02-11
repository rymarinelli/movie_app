#!/usr/bin/env python
import time
import requests
from kubernetes import client, config

# ---- Configuration ----
# URL to access your application (e.g., the Minikube tunnel URL or NodePort URL)
APP_URL = "http://127.0.0.1:37603" 

# Kubernetes namespace and label selector for your app's pods
NAMESPACE = "default"
LABEL_SELECTOR = "app=movie-app"  # Ensure your pods have a matching label

# Interval (in seconds) between each monitoring check
MONITOR_INTERVAL = 10

# ---- Monitoring Functions ----

def monitor_http():
    """
    Sends an HTTP GET request to the app endpoint and returns the response time and status code.
    """
    try:
        start_time = time.time()
        response = requests.get(APP_URL)
        elapsed = time.time() - start_time
        status = response.status_code
        return elapsed, status
    except Exception as e:
        print("Error monitoring HTTP:", e)
        return None, None

def monitor_k8s():
    """
    Uses the Kubernetes API to list pods in the specified namespace that match the given label.
    Returns a list of dictionaries with pod name, IP, and status.
    """
    try:
        # Load kubeconfig (for local development) or use in-cluster config if running inside Kubernetes
        config.load_kube_config()
        v1 = client.CoreV1Api()
        pod_list = v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=LABEL_SELECTOR)
        pods_info = []
        for pod in pod_list.items:
            pods_info.append({
                "name": pod.metadata.name,
                "ip": pod.status.pod_ip,
                "status": pod.status.phase
            })
        return pods_info
    except Exception as e:
        print("Error monitoring Kubernetes pods:", e)
        return []

# ---- Main Monitoring Loop ----

def main():
    while True:
        # Monitor the application via HTTP
        http_elapsed, http_status = monitor_http()
        if http_elapsed is not None:
            print(f"HTTP Response: {http_status} in {http_elapsed:.2f} seconds")
        else:
            print("HTTP monitoring failed.")

        # Monitor Kubernetes pods
        pods = monitor_k8s()
        print("Kubernetes Pods:")
        if pods:
            for pod in pods:
                print(f" - {pod['name']}: {pod['status']} (IP: {pod['ip']})")
        else:
            print("No pods found or error fetching pods.")

        # Aggregate metrics (could be used as input for an RL agent)
        metrics = {
            "http_response_time": http_elapsed,
            "http_status": http_status,
            "pod_count": len(pods),
            "pods": pods
        }

        print("Aggregated Metrics:", metrics)
        print("-" * 40)
        
        # Wait before the next check
        time.sleep(MONITOR_INTERVAL)

if __name__ == "__main__":
    main()
