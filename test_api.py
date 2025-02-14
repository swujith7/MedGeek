import requests

response = requests.get("http://127.0.0.1:8000/plot")
data = response.json()

print(data["plot"])  # This will print the Base64-encoded image
