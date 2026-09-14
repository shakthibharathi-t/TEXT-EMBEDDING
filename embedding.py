from sentence_transformers import SentenceTransformer
import pandas as pd

# Read text data
with open("Dataset/texts.txt", "r", encoding="utf-8") as file:
    texts = [line.strip() for line in file if line.strip()]

print("Number of texts:", len(texts))

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")

# Generate embeddings
embeddings = model.encode(texts)

print("Embeddings generated successfully!")
print("Embedding shape:", embeddings.shape)

# Save embeddings
df = pd.DataFrame(embeddings)
df.insert(0, "Text", texts)

df.to_csv("Dataset/embeddings.csv", index=False)

print("Embeddings saved successfully!")