from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from PIL import Image
import torch
import os, json
from tqdm import tqdm

# --- Cấu hình model và Qdrant ---
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "tropical_disease_cases"

# --- Load model ---
model = SentenceTransformer(TEXT_MODEL, device="cpu")  # nếu có GPU, đổi thành "cuda"

# --- Kết nối Qdrant ---
client = QdrantClient(url=QDRANT_URL)

# --- Tạo collection nếu chưa có ---
collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in collections:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
    )
    print(f"✅ Created collection: {COLLECTION_NAME}")
else:
    print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists")

# --- Nạp dữ liệu từ dataset_ready.jsonl ---
dataset_path = "dataset_ready.jsonl"
points = []

with open(dataset_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Embedding cases"):
        item = json.loads(line)
        case_id = item["id"]
        text_input = item["text_input"]
        label = item.get("label", "")
        image_paths = item.get("images", [])

        # Encode text
        text_vec = model.encode(text_input, convert_to_tensor=True)

        # Encode images (lấy trung bình nếu có nhiều ảnh)
        img_vecs = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img_vec = model.encode(img, convert_to_tensor=True)
                img_vecs.append(img_vec)
            except Exception as e:
                print(f"⚠️ Lỗi ảnh {img_path}: {e}")

        if img_vecs:
            image_vec = torch.stack(img_vecs).mean(dim=0)
        else:
            image_vec = torch.zeros_like(text_vec)

        # Combine (average text + image)
        combined_vec = (text_vec + image_vec) / 2
        combined_vec = combined_vec.cpu().tolist()

        # Tạo record
        point = models.PointStruct(
            id=int(case_id.split("-")[-1]),
            vector=combined_vec,
            payload={
                "id": case_id,
                "text": text_input,
                "label": label,
                "images": image_paths
            }
        )
        points.append(point)

# --- Upload lên Qdrant ---
client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"✅ Uploaded {len(points)} cases to Qdrant collection '{COLLECTION_NAME}'")
