import os
import json
import glob
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "tropical_disease_cases_mm"  # mm = multimodal
JSON_FOLDER = "json-output"
IMAGE_FOLDER = "images"

# --- INIT MODEL & CLIENT ---
model = SentenceTransformer("clip-ViT-B-32")
client = QdrantClient(url=QDRANT_URL)

# --- CREATE COLLECTION ---
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
    )
    print(f"✅ Created collection: {COLLECTION_NAME}")
else:
    print(f"⚠️ Collection '{COLLECTION_NAME}' already exists — will append new data.")

# --- FUNCTION: Upload one case ---
def upload_case(case_id, text_data, image_paths):
    points = []

    # Encode text block
    text = " ".join([f"{k}: {v}" for k, v in text_data.items()])
    text_vector = model.encode(text).tolist()
    points.append(
        models.PointStruct(
            id=int(f"{case_id}001"),
            vector=text_vector,
            payload={"type": "text", "case_id": case_id, **text_data},
        )
    )

    # Encode images if exist
    for idx, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            img_vector = model.encode(img).tolist()
            points.append(
                models.PointStruct(
                    id=int(f"{case_id}{idx+10}"),
                    vector=img_vector,
                    payload={
                        "type": "image",
                        "case_id": case_id,
                        "image_path": img_path,
                    },
                )
            )
        except Exception as e:
            print(f"⚠️ Failed to process image {img_path}: {e}")

    client.upsert(collection_name=COLLECTION_NAME, points=points)


# --- MAIN ---
def main():
    json_files = sorted(glob.glob(os.path.join(JSON_FOLDER, "*.json")))
    print(f"Found {len(json_files)} JSON files.")

    for json_file in tqdm(json_files, desc="Embedding cases"):
        case_id = int(os.path.basename(json_file).split("Case-")[1].split(".")[0])
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Find images for this case
        case_folder = os.path.join(IMAGE_FOLDER, f"Case-{case_id}")
        image_paths = glob.glob(os.path.join(case_folder, "*.jpg")) + \
                      glob.glob(os.path.join(case_folder, "*.png"))

        upload_case(case_id, data, image_paths)

    print(f"✅ Uploaded multimodal data to Qdrant collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
