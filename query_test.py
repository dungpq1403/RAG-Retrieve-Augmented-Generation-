from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# --- Qdrant connection settings ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "tropical_disease_cases"

# --- Load CLIP model for text embeddings ---
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

# --- Connect to Qdrant client ---
client = QdrantClient(url=QDRANT_URL)

# --- User input ---
query = input("Enter a description of the disease case to search for: ")

# --- Generate embedding for the query ---
query_vector = model.encode(query).tolist()

# --- Search for the 3 most similar cases ---
results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=3
)

# --- Display results ---
print("\n🩺 Top 3 most similar cases found:\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r.payload['id']} — {r.payload.get('label', 'Unknown disease')}")
    print(f"   Similarity score: {r.score:.3f}")
    print(f"   Description: {r.payload['text'][:300]}...\n")
