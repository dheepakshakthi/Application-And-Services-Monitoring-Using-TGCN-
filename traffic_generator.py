import math
import random
import time

import requests

TARGET_URL = "http://localhost:8000/"


def generate_traffic():
    print("Starting traffic generation...")
    hour = 0
    while True:
        is_night = hour < 6 or hour > 20
        time_of_day = "Night" if is_night else "Day"

        # Normal Conditions
        if not is_night:
            base_users = random.randint(200, 500)
            high_resource = True  # High usage normal during daytime
        else:
            base_users = random.randint(10, 50)
            high_resource = False  # Stable usage normal at night

        users = base_users
        inject_crash = False

        # Abnormalities
        if is_night and random.random() < 0.20:
            print(f"[{hour:02d}:00] Anomaly: High traffic spike at night!")
            users += random.randint(300, 600)
            high_resource = True

        if not is_night and random.random() < 0.15:
            print(f"[{hour:02d}:00] Anomaly: Extreme resource exhaustion during day!")
            users += random.randint(100, 300)
            high_resource = True

        # Crash injection randomly triggered by heavy load or usage
        if users > 300 or high_resource:
            if random.random() < 0.25:  # 25% chance to crash if system is under stress
                print(
                    f"[{hour:02d}:00] Anomaly: Service crash injected due to high load!"
                )
                inject_crash = True

        print(
            f"--- Simulating hour {hour:02d} ({time_of_day}) | Users: {users} | High Res: {high_resource} | Crash: {inject_crash} ---"
        )

        params = {
            "users": users,
            "inject_crash": inject_crash,
            "high_resource": high_resource,
            "simulated_hour": hour,
        }

        num_requests = max(
            1, users // 50
        )  # Scale down requests to prevent infinite loops
        for i in range(num_requests):
            try:
                resp = requests.get(TARGET_URL, params=params, timeout=10)
                if resp.status_code == 200:
                    print(f"  -> [{i + 1}/{num_requests}] Request OK.")
                else:
                    print(
                        f"  -> [{i + 1}/{num_requests}] Request Error: HTTP {resp.status_code}"
                    )
            except Exception as e:
                print(f"  -> [{i + 1}/{num_requests}] Request Failed: {e}")
            time.sleep(random.uniform(0.1, 0.3))

        hour = (hour + 1) % 24
        time.sleep(1)


if __name__ == "__main__":
    generate_traffic()
