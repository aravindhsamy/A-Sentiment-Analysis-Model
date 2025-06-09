import requests

url = "http://localhost:1234/invocations"
data = {"inputs": ["This movie was fantastic!"]}

response = requests.post(url, json=data)
print(response.json())
