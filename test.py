import requests, base64

with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:11436/api/embed", json={"input": img_b64})
print(resp.status_code, len(resp.json()["embeddings"][0]))