import os
import numpy as np
from huggingface_hub import InferenceClient
from openai import OpenAI

# Hugging Face client for embeddings
hf_client = InferenceClient(
    api_key=os.environ["HF_TOKEN"]
)

# OpenAI-compatible client for LLM
llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

# Read document
with open("placement.txt", "r") as file:
    text = file.read()

# Split into chunks
chunks = text.split("\n\n")

# Create vectors for chunks
chunk_vectors = []

for chunk in chunks:
    vector = hf_client.feature_extraction(
        chunk,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_vectors.append(vector)

# Keep asking questions
while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # Convert question into a vector
    question_vector = hf_client.feature_extraction(
        question,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Find similarity scores
    question_vector = np.array(question_vector)

    scores = []

    for i, vector in enumerate(chunk_vectors):

        vector = np.array(vector)

        similarity = np.dot(vector, question_vector) / (
            np.linalg.norm(vector) * np.linalg.norm(question_vector)
        )

        print("Chunk", i + 1, "similarity:", similarity)

        scores.append((similarity, chunks[i]))

    # Sort chunks from highest similarity to lowest
    scores.sort(reverse=True)

    # Get the best chunk
    # Get the top 2 chunks
    top_k = 2
    threshold = 0.70

    top_chunks = [
    (score, chunk)
    for score, chunk in scores[:top_k]
    if score >= threshold
]
    if not top_chunks:
        print("\nNo relevant information found.")
        continue

    # Check similarity threshold
    if top_chunks and top_chunks[0][0] >= 0.70:

        print("\nTop chunks:")

        for score, chunk in top_chunks:
            print("\nSimilarity:", score)
            print(chunk)

    else:
        print("\nNo relevant information found.")
        continue

    # Send retrieved information to LLM
    context = "\n\n".join(chunk for score, chunk in top_chunks)
    response = llm_client.chat.completions.create(
    model="openai/gpt-oss-120b:fastest",
    messages=[
        {
            "role": "user",
            "content": f"""
Answer the question using ONLY the information provided below.

Do not use your own knowledge.
Do not add information that is not explicitly present in the provided information.

If the answer is not present in the information, say:
"The information is not available in the provided document."

Information:
{context}

Question:
{question}
"""
        }
    ]
)

    print("\nLLM Answer:")
    print(response.choices[0].message.content)
