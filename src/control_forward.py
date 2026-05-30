import requests
import time

control_url = "http://172.20.10.5:5000/control"

def forward(duration):
    print(f"前进 {duration} 秒...")
    response = requests.post(control_url, json={'command': "RIGHT"})
    print(f"响应: {response.status_code}")
    time.sleep(duration)
    response = requests.post(control_url, json={'command': "STOP"})
    print(f"停止，响应: {response.status_code}")

if __name__ == "__main__":
    forward(5)
