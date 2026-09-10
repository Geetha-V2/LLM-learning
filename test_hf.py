import os

token = os.getenv("HF_TOKEN")

if token:
    print("Hugging Face token found!")
else:
    print("Token not found!")