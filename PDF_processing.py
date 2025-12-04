import fitz
import json
import os
import glob
import re
from google import genai

# --- Cấu hình Gemini ---
client = genai.Client(api_key="AIzaSyDDvw6S5PQVCYmmRSxZEP97ZgWnbzvD1PA")
MODEL = "gemini-2.5-flash"

# --- Danh sách key bắt buộc trong JSON ---
REQUIRED_KEYS = [
    "patient_information",
    "chief_complaint",
    "history_of_present_illness",
    "exposure_and_epidemiology",
    "vitals",
    "physical_exam",
    "labs_and_diagnostics",
    "differential_diagnosis",
    "management_and_clinical_course",
    "final_diagnosis",
    "disease_name_short"
]

# --- Hàm trích text từ PDF ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# --- Hàm trích ảnh từ PDF ---
def extract_images_from_pdf(pdf_path, case_folder):
    doc = fitz.open(pdf_path)
    os.makedirs(case_folder, exist_ok=True)
    img_filenames = []

    for page_index, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            img_filename = f"page{page_index+1}_img{img_index+1}.{image_ext}"
            img_path = os.path.join(case_folder, img_filename)

            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)

            img_filenames.append(img_filename)

    return img_filenames  # ← Trả về danh sách tên ảnh

# --- Hàm gọi Gemini để trích thông tin ---
def extract_case_info(text):
    system_prompt = (
        "You are a medical text extraction model specialized in tropical infectious disease case reports. "
        "Extract structured patient information from the input text according to the given JSON schema."
    )
    schema = """
Return JSON in this format:
{
  "patient_information": "...",
  "chief_complaint": "...",
  "history_of_present_illness": "...",
  "exposure_and_epidemiology": "...",
  "vitals": "...",
  "physical_exam": "...",
  "labs_and_diagnostics": "...",
  "differential_diagnosis": "...",
  "management_and_clinical_course": "...",
  "final_diagnosis": "...",
  "disease_name_short": "..."
}
If a section is not present, use "Not mentioned".
"""

    prompt = (
        f"{system_prompt}\n\n{schema}\n\n"
        "Return ONLY the JSON object for the extracted data. "
        "Do not include any explanatory text or markdown code blocks.\n\n"
        f"TEXT:\n{text[:8000]}"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    content = response.text

    # Làm sạch JSON nếu model trả về markdown
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    if match:
        json_string = match.group(1)
    else:
        json_string = content

    json_string = json_string.strip()

    try:
        data = json.loads(json_string)
    except Exception as e:
        print("⚠️ Không parse JSON được:", e)
        data = {"raw_output": content}

    # ✅ Đảm bảo luôn đủ 10 keys
    for key in REQUIRED_KEYS:
        if key not in data:
            data[key] = "Not mentioned"

    return data

# --- Hàm xử lý toàn bộ folder ---
def process_all_pdfs(pdf_folder="PDF-cases", json_folder="json-output", image_folder="image-output"):
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_paths:
        print(f"❌ Không tìm thấy tệp PDF nào trong thư mục: {pdf_folder}")
        return

    for pdf in pdf_paths:
        print("📄 Processing:", pdf)
        try:
            # 1️⃣ Trích text
            text = extract_text_from_pdf(pdf)
            # 2️⃣ Gọi Gemini để lấy JSON
            data = extract_case_info(text)

            # 3️⃣ Tạo tên file và thư mục ảnh tương ứng
            basename = os.path.basename(pdf)
            no_ext = os.path.splitext(basename)[0]
            case_num = no_ext.split("---")[0]
            json_filename = f"Case-{case_num}.json"
            case_img_folder = os.path.join(image_folder, f"Case-{case_num}")

            # 4️⃣ Trích ảnh
            image_list = extract_images_from_pdf(pdf, case_img_folder)
            print(f"🖼️ {len(image_list)} ảnh được trích xuất đến: {case_img_folder}")

            # 5️⃣ Gắn danh sách ảnh vào JSON
            data["images"] = image_list if image_list else []

            # 6️⃣ Lưu JSON
            out_path = os.path.join(json_folder, json_filename)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved: {out_path}")

        except Exception as e:
            print(f"🔥 Lỗi xử lý tệp {pdf}: {e}")
            with open("error_log.txt", "a", encoding="utf-8") as log:
                log.write(f"{pdf} → {e}\n")

if __name__ == "__main__":
    process_all_pdfs()
