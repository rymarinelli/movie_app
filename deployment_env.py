import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class DeploymentEnv(gym.Env):
    """
    Environment optimizes the configuration of a minikube deployment.
    Agent takes actions to scale the number of pods in the 'movie-app' deployment based
    on the observed performance of the recommender service.

    Actions:
      0: Scale down (reduce replicas)
      1: Do nothing
      2: Scale up (increase replicas)

    Observations:
      A 2-element vector [response_time, pod_count] where:
        - response_time is measured in seconds (0.0 to 10.0, with higher values penalized)
        - pod_count is the current number of pods (from 1 to 10)

    Reward:
      Base reward = -response_time - 0.1 * pod_count
      Additional penalties are applied periodically:
         - A stress penalty if the service fails under a burst of requests.
         - A resource penalty if too many pods are allocated.
    """
    
    def __init__(self, base_url=None, realistic_usage=True, wait_time=1):
        """
        :param base_url: Base URL for the service (if None, auto-detect).
        :param realistic_usage: If True, simulate realistic POST requests.
        :param wait_time: Time (in seconds) to wait after scaling operations.
                          Reduce this for faster training.
        """
        super(DeploymentEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 1], dtype=np.float32),
            high=np.array([10.0, 10], dtype=np.float32),
            dtype=np.float32
        )
        self.base_url = base_url if base_url is not None else self.get_service_url()
        if self.base_url is None:
            raise Exception("Unable to determine a reachable base_url for the service.")
        if realistic_usage and not self.base_url.endswith("/recommend"):
            self.base_url = self.base_url.rstrip("/") + "/recommend"
        print(f"[INIT] Using base_url: {self.base_url}")
        self.realistic_usage = realistic_usage
        self.current_replicas = self.get_pod_count()
        print(f"[INIT] Initial pod count: {self.current_replicas}")
        self.step_counter = 0
        self.stress_test_interval = 3
        self.wait_time = wait_time #Might have to decrease wait time in training 

    def get_pod_count(self):
        try:
            output = subprocess.check_output(
                ["kubectl", "get", "pods", "-l", "app=movie-app", "--no-headers"],
                stderr=subprocess.DEVNULL
            )
            pod_lines = output.decode("utf-8").splitlines()
            pod_count = len(pod_lines)
            print(f"[GET_POD_COUNT] Retrieved pod count: {pod_count}")
            return pod_count
        except Exception as e:
            print(f"[GET_POD_COUNT] Error retrieving pod count: {e}")
            return 1

    def get_service_url(self):
        try:
            output = subprocess.check_output(
                ["kubectl", "get", "svc", "movie-app-service", "-o", "json"],
                stderr=subprocess.DEVNULL
            )
            svc = json.loads(output)
            node_port = svc["spec"]["ports"][0]["nodePort"]
            minikube_ip = subprocess.check_output(["minikube", "ip"]).decode("utf-8").strip()
            service_url = f"http://{minikube_ip}:{node_port}"
            print(f"[GET_SERVICE_URL] Detected service URL: {service_url}")
            return service_url
        except Exception as e:
            print(f"[GET_SERVICE_URL] Error retrieving service URL: {e}")
            return None

    def get_response_time(self):
        session = requests.Session()
        session.headers.update({"Connection": "close"})
        max_attempts = 3
        backoff = 1
        payload = {}
        if self.realistic_usage:
            selection = np.random.randint(0, 2, size=50).tolist()
            payload = {"selection": selection}
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"[GET_RESPONSE_TIME] Attempt {attempt} sending request to {self.base_url}")
                start = time.time()
                if self.realistic_usage:
                    response = session.post(self.base_url, json=payload, timeout=5)
                else:
                    response = session.get(self.base_url, timeout=5)
                elapsed = time.time() - start
                if response.status_code != 200:
                    print(f"[GET_RESPONSE_TIME] Attempt {attempt}: Non-200 status code: {response.status_code}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                print(f"[GET_RESPONSE_TIME] Successful connection in {elapsed:.3f} seconds")
                return elapsed
            except requests.exceptions.RequestException as e:
                print(f"[GET_RESPONSE_TIME] Attempt {attempt}: Error connecting to service: {e}")
                time.sleep(backoff)
                backoff *= 2
        print("[GET_RESPONSE_TIME] All attempts failed. Returning penalty value of 10.0 seconds")
        return 10.0

    def stress_test(self):
        n_requests = np.random.randint(50, 150)  # burst size
        max_concurrent_requests = np.random.randint(50, 150)  #  parallel threads
        responses = []
        failures = 0

        print(f"[STRESS_TEST] Running {n_requests} concurrent requests with max {max_concurrent_requests} threads.")

        def send_request():
            payload = {}
            if self.realistic_usage:
                selection = np.random.randint(0, 2, size=50).tolist()
                payload = {"selection": selection}

            try:
                start = time.time()
                if self.realistic_usage:
                    r = requests.post(self.base_url, json=payload, timeout=5)
                else:
                    r = requests.get(self.base_url, timeout=5)
                elapsed = time.time() - start

                if r.status_code != 200:
                    print(f"[STRESS_TEST] Request failed with status {r.status_code}")
                    return 10.0, True  # Failure, use max penalty

                return elapsed, False  # Success, return response time

            except Exception as e:
                print(f"[STRESS_TEST] Error: {e}")
                return 10.0, True  # Failure case

        # Run parallel stress test
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            future_to_request = {executor.submit(send_request): i for i in range(n_requests)}

            for future in as_completed(future_to_request):
                response_time, failed = future.result()
                responses.append(response_time)
                if failed:
                    failures += 1

        avg_response = np.mean(responses)
        success_rate = (n_requests - failures) / n_requests
        penalty = 0

        # Apply penalties for failures and slow responses
        if success_rate < 1.0:
            penalty += 5 * (1.0 - success_rate)
        threshold = 2.0  # Response time threshold for penalty
        if avg_response > threshold:
            penalty += (avg_response - threshold) * 2

        print(f"[STRESS_TEST] Avg response: {avg_response:.3f}, Success rate: {success_rate:.2f}, Penalty: {penalty}")

        return penalty

    def resource_test(self):
        desired_max = 5
        penalty = 0
        if self.current_replicas > desired_max:
            penalty = (self.current_replicas - desired_max) * 2
            print(f"[RESOURCE_TEST] Too many replicas! Current: {self.current_replicas}. Penalty: {penalty}")
        else:
            print(f"[RESOURCE_TEST] Resource allocation within limits. Current replicas: {self.current_replicas}")
        return penalty

    def scale_deployment(self, replicas):
        try:
            print(f"[SCALE_DEPLOYMENT] Scaling deployment/movie-app to {replicas} replicas.")
            subprocess.run(
                ["kubectl", "scale", "deployment/movie-app", f"--replicas={replicas}"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[SCALE_DEPLOYMENT] Successfully scaled to {replicas} replicas.")
        except Exception as e:
            print(f"[SCALE_DEPLOYMENT] Error scaling deployment: {e}")

    def step(self, action):
        print(f"[STEP] Received action: {action}")

        # scaling action
        if action == 0:
            self.current_replicas = max(1, self.current_replicas - 1)
            print(f"[STEP] Scaling down. New replica count: {self.current_replicas}")
        elif action == 2:
            self.current_replicas += 1
            print(f"[STEP] Scaling up. New replica count: {self.current_replicas}")
        else:
            print(f"[STEP] No scaling action taken. Replica count remains: {self.current_replicas}")

        self.scale_deployment(self.current_replicas)

    
        print(f"[STEP] Waiting {self.wait_time} second(s) for deployment update...")
        time.sleep(self.wait_time)

  
        response_time = self.get_response_time()
        pod_count = self.get_pod_count()
        state = np.array([response_time, pod_count], dtype=np.float32)

        #reward
        reward = -response_time - 0.1 * pod_count
        self.step_counter += 1

        # Apply stress test every few steps
        stress_penalty = 0
        if self.step_counter % self.stress_test_interval == 0:
            stress_penalty = self.stress_test()
            reward -= stress_penalty
            print(f"[STEP] Applied stress penalty: {stress_penalty}")

        # Apply resource penalty
        resource_penalty = self.resource_test()
        reward -= resource_penalty
        print(f"[STEP] Applied resource penalty: {resource_penalty}")

        #Auto-Scaling
        if stress_penalty > 0:
            increase_by = max(1, stress_penalty // 2)  # Scale up dynamically
            new_replicas = min(self.current_replicas + increase_by, 10)  # Prevent excessive scaling

            if new_replicas > self.current_replicas:
                print(f"[STEP] Severe stress detected! Scaling from {self.current_replicas} → {new_replicas} replicas.")
                self.current_replicas = new_replicas
                self.scale_deployment(self.current_replicas)
                time.sleep(self.wait_time)  # Allow system to stabilize
                pod_count = self.get_pod_count()  # Update pod count after scaling

        print(f"[STEP] New state: {state}, Reward: {reward}")

        # Termination conditions (modify if needed)
        terminated = False
        truncated = False
        info = {}

        return state, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        print("[RESET] Resetting environment to initial state (1 replica).")
        self.current_replicas = 1
        self.scale_deployment(self.current_replicas)
        print(f"[RESET] Waiting {self.wait_time} second(s) for deployment update...")
        time.sleep(self.wait_time)
        response_time = self.get_response_time()
        pod_count = self.get_pod_count()
        state = np.array([response_time, pod_count], dtype=np.float32)
        print(f"[RESET] Initial state: {state}")
        self.step_counter = 0
        info = {}
        return state, info

    def close(self):
        print("[CLOSE] Closing environment resources.")
        pass

