#!/usr/bin/env python3
"""
rag_with_gemini.py

RAG pipeline dùng:
- Qdrant (retriever, vector DB)
- SentenceTransformers (CLIP-ViT-B-32) để embed câu truy vấn
- Gemini API để sinh câu trả lời chẩn đoán y khoa

👉 Bạn chỉ cần đổi GEMINI_API_KEY bên dưới khi hết lượt.
"""

import os
import sys
import textwrap
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai

# -----------------------
# 🔧 Cấu hình chính
# -----------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "tropical_disease_cases_mm"
EMBED_MODEL_NAME = "clip-ViT-B-32"
GEMINI_MODEL = "gemini-2.5-flash"

# ⚠️ 👉 Thêm key thủ công ở đây
GEMINI_API_KEY = "AIzaSyDDvw6S5PQVCYmmRSxZEP97ZgWnbzvD1PA"

if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
    print("❌ Bạn chưa thêm GEMINI_API_KEY. Vui lòng thêm key trong file này rồi chạy lại.")
    sys.exit(1)

# -----------------------
# 🚀 Khởi tạo client
# -----------------------
print("🔗 Connecting to Qdrant and Gemini...")
qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------
# ⚙️ Hàm tiện ích
# -----------------------
def retrieve_top_k(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Lấy top-k kết quả tương tự từ Qdrant."""
    q_vec = embedder.encode(query).tolist()
    hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=top_k)
    results = []
    for h in hits:
        payload = h.payload or {}
        results.append({
            "score": getattr(h, "score", None),
            "payload": payload
        })
    return results

def build_context_snippets(hits: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    """Ghép các đoạn mô tả case thành context cho prompt."""
    parts = []
    total_len = 0
    for i, hit in enumerate(hits, start=1):
        payload = hit["payload"]
        case_id = payload.get("case_id") or payload.get("id") or f"unknown-{i}"
        text_field = payload.get("text") or payload.get("final_diagnosis") or payload.get("management_and_clinical_course") or ""
        snippet = text_field.strip().replace("\n", " ")
        if len(snippet) > 600:
            snippet = snippet[:590] + " ..."
        entry_lines = [f"--- Case {case_id} (score={hit.get('score'):.4f}):"]
        if snippet:
            entry_lines.append(f"Text snippet: {snippet}")
        if payload.get("image_path"):
            entry_lines.append(f"Image path: {payload['image_path']}")
        elif payload.get("images"):
            imgs = payload["images"]
            if isinstance(imgs, list) and imgs:
                entry_lines.append(f"Image path: {imgs[0]}")
        entry_text = " ".join(entry_lines)
        if total_len + len(entry_text) > max_chars:
            break
        parts.append(entry_text)
        total_len += len(entry_text)
    return "\n\n".join(parts) if parts else "No relevant context found."

def build_prompt(context: str, question: str) -> str:
    """Tạo prompt hoàn chỉnh gửi đến Gemini."""
    system = (
        "You are a medical assistant specialized in tropical infectious diseases. "
        "Use the context from clinical case reports below to suggest possible diagnoses and management plans. "
        "If uncertain, state what information is missing."
    )
    prompt = textwrap.dedent(f"""
    SYSTEM INSTRUCTION:
    {system}

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    Please answer concisely in 3 parts:
    1. Summary (1–2 sentences)
    2. Likely diagnosis (brief reasoning)
    3. Suggested next diagnostic steps or management.
    """).strip()
    return prompt

def call_gemini(prompt: str) -> str:
    """Gửi prompt đến Gemini và trả về phản hồi."""
    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Lỗi khi gọi Gemini API: {e}"

# -----------------------
# 💬 Chế độ tương tác
# -----------------------
def interactive_loop():
    print("\n=== 🧠 Tropical Disease Diagnosis (RAG + Gemini) ===")
    print("Nhập mô tả triệu chứng hoặc câu hỏi. Gõ 'exit' để thoát.")
    while True:
        question = input("\n🔎 Câu hỏi: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("👋 Tạm biệt!")
            break

        print("\n🔍 Đang truy xuất dữ liệu liên quan từ Qdrant...")
        hits = retrieve_top_k(question)
        if not hits:
            print("❌ Không tìm thấy case nào phù hợp.")
            continue

        print(f"✅ Tìm thấy {len(hits)} case tương tự.")
        context = build_context_snippets(hits)
        prompt = build_prompt(context, question)

        print("\n🤖 Đang gọi Gemini để sinh câu trả lời...")
        answer = call_gemini(prompt)
        print("\n================= 🩺 KẾT QUẢ =================\n")
        print(answer)
        print("\n==============================================")

if __name__ == "__main__":
    try:
        interactive_loop()
    except KeyboardInterrupt:
        print("\n🚪 Dừng bởi người dùng. Tạm biệt!")
