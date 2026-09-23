import faiss
import numpy as np

# Three simple vectors
vectors = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
], dtype="float32")

# Create FAISS index
index = faiss.IndexFlatL2(2)

# Add vectors to FAISS
index.add(vectors)

print("Number of vectors:", index.ntotal)

# Search for a vector
query = np.array([
    [1, 0]
], dtype="float32")

distances, indices = index.search(query, 2)

print("Distances:", distances)
print("Indices:", indices)