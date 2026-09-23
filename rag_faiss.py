import faiss
import numpy as np
import os
from huggingface_hub import InferenceClient

# Hugging Face client
hf_client = InferenceClient(
    api_key=os.environ["HF_TOKEN"]
)

# Read document
with open("placement.txt", "r") as file:
    text = file.read()

# Split document into chunks
chunks = text.split("\n\n")

# Create embeddings
chunk_vectors = []

for chunk in chunks:

    vector = hf_client.feature_extraction(
        chunk,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_vectors.append(vector)

# Convert vectors to NumPy array
chunk_vectors = np.array(chunk_vectors, dtype="float32")

print("Number of chunks:", len(chunks))
print("Vector shape:", chunk_vectors.shape)

# Create FAISS index
index = faiss.IndexFlatL2(384)

# Add vectors to FAISS
index.add(chunk_vectors)

print("Vectors stored in FAISS:", index.ntotal)
# Ask a question
question = input("\nYou: ")

# Convert question into a vector
question_vector = hf_client.feature_extraction(
    question,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert to NumPy array
question_vector = np.array(
    [question_vector],
    dtype="float32"
)

# Search FAISS
distances, indices = index.search(question_vector, 1)

# Get best result
best_index = indices[0][0]
best_distance = distances[0][0]

print("\nBest chunk:")
print(chunks[best_index])

print("\nDistance:", best_distance)
print("Index:", best_index)