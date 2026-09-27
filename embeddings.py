import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("Dataset/generated_text.csv")

print("Dataset loaded successfully!")
print(df)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get sentences
sentences = df["text"].tolist()

# Generate embeddings
embeddings = model.encode(sentences)

print("\nEmbeddings generated!")
print("Embedding shape:", embeddings.shape)

# Save embeddings
embedding_data = pd.DataFrame(embeddings)
embedding_data.insert(0, "id", df["id"])

embedding_data.to_csv(
    "Dataset/embeddings.csv",
    index=False
)

print("embeddings.csv created!")

# Calculate cosine similarity
similarity = cosine_similarity(embeddings)

# Create similarity score list
results = []

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        results.append([
            df.iloc[i]["id"],
            df.iloc[j]["id"],
            similarity[i][j]
        ])

# Convert results to DataFrame
similarity_df = pd.DataFrame(
    results,
    columns=[
        "sentence_1_id",
        "sentence_2_id",
        "similarity_score"
    ]
)

# Sort scores
similarity_df = similarity_df.sort_values(
    "similarity_score",
    ascending=False
)

# Save similarity scores
similarity_df.to_csv(
    "Dataset/similarity_scores.csv",
    index=False
)

print("similarity_scores.csv created!")

# Create sentence pairs
pairs = []

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        pairs.append([
            df.iloc[i]["id"],
            sentences[i],
            df.iloc[j]["id"],
            sentences[j]
        ])

pairs_df = pd.DataFrame(
    pairs,
    columns=[
        "sentence_1_id",
        "sentence_1",
        "sentence_2_id",
        "sentence_2"
    ]
)

pairs_df.to_csv(
    "Dataset/sentence_pairs.csv",
    index=False
)

print("sentence_pairs.csv created!")

# Display top 5 similar sentences
print("\nTop 5 Similar Sentence Pairs:")

for _, row in similarity_df.head(5).iterrows():

    s1 = df[df["id"] == row["sentence_1_id"]]["text"].values[0]
    s2 = df[df["id"] == row["sentence_2_id"]]["text"].values[0]

    print("\nSentence 1:", s1)
    print("Sentence 2:", s2)
    print("Similarity Score:", round(row["similarity_score"], 4))

print("\nProject completed successfully!")