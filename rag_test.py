import os
import numpy as np
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

# Read the document
with open("placement.txt", "r") as file:
    text = file.read()

# Split document into chunks
chunks = text.split("\n\n")

print("Number of chunks:", len(chunks))

# Create a vector for every chunk
chunk_vectors = []

for chunk in chunks:
    vector = client.feature_extraction(
        chunk,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_vectors.append(vector)

print("Number of chunk vectors:", len(chunk_vectors))

# User's question
question = "What is the minimum CGPA required by ABC Company?"

# Convert question into a vector
question_vector = client.feature_extraction(
    question,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print("Question vector length:", len(question_vector))

# Find the most similar chunk
best_score = -1
best_chunk = ""

question_vector_np = np.array(question_vector)

for i, vector in enumerate(chunk_vectors):

    vector = np.array(vector)

    similarity = np.dot(vector, question_vector_np) / (
        np.linalg.norm(vector) * np.linalg.norm(question_vector_np)
    )

    print("Chunk", i + 1, "similarity:", similarity)

    if similarity > best_score:
        best_score = similarity
        best_chunk = chunks[i]

print("\nBest chunk:")
print(best_chunk)

print("\nSimilarity score:", best_score)