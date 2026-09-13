import os
import requests

token = os.getenv("HF_TOKEN")

url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

headers = {
    "Authorization": f"Bearer {token}"
}

data = {
    "inputs": {
        "source_sentence": "I love learning Java and LLMs.",
        "sentences": [
            "I enjoy studying Java and LLMs.",
            "The weather is very hot today."
        ]
    }
}

response = requests.post(url, headers=headers, json=data)

print(response.json())