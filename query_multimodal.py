import os
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- CONFIG ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "tropical_disease_cases_mm"

# --- INIT ---
model = SentenceTransformer("clip-ViT-B-32")
client = QdrantClient(url=QDRANT_URL)


def search_by_text(query_text, top_k=5):
    """Search by text description"""
    query_vector = model.encode(query_text).tolist()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    print(f"\n🔍 Query: {query_text}\n")
    for i, hit in enumerate(hits, start=1):
        payload = hit.payload
        print(f"{i}. Type: {payload.get('type')}, Case ID: {payload.get('case_id')}")
        if payload.get("type") == "text":
            print(f"   Disease: {payload.get('disease_name_short', 'N/A')}")
            print(f"   Final Diagnosis: {payload.get('final_diagnosis', 'N/A')[:150]}...")
        elif payload.get("type") == "image":
            print(f"   Image path: {payload.get('image_path')}")
        print(f"   Score: {hit.score:.4f}\n")


def search_by_image(image_path, top_k=5):
    """Search by image file"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    image = Image.open(image_path).convert("RGB")
    query_vector = model.encode(image).tolist()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    print(f"\n🖼️ Query Image: {image_path}\n")
    for i, hit in enumerate(hits, start=1):
        payload = hit.payload
        print(f"{i}. Type: {payload.get('type')}, Case ID: {payload.get('case_id')}")
        if payload.get("type") == "text":
            print(f"   Disease: {payload.get('disease_name_short', 'N/A')}")
            print(f"   Final Diagnosis: {payload.get('final_diagnosis', 'N/A')[:150]}...")
        elif payload.get("type") == "image":
            print(f"   Image path: {payload.get('image_path')}")
        print(f"   Score: {hit.score:.4f}\n")


def main():
    print("=== Multimodal Search ===")
    print("1️⃣  Search by text")
    print("2️⃣  Search by image")
    choice = input("Choose mode (1 or 2): ").strip()

    if choice == "1":
        query = input("\nEnter your text description: ")
        search_by_text(query)
    elif choice == "2":
        img_path = input("\nEnter the image file path: ")
        search_by_image(img_path)
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
