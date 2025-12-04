import fitz
import json
import os
import glob
import re # Đã thêm thư viện Re (Regular Expression) để xử lý chuỗi

# Nếu dùng Gemini SDK:
from google import genai
client = genai.Client(api_key="AIzaSyDDvw6S5PQVCYmmRSxZEP97ZgWnbzvD1PA")
MODEL = "gemini-2.5-flash"

# Nếu bạn vẫn muốn dùng OpenAI (nếu bạn có key OpenAI), giữ phần này:
# from openai import OpenAI
# client = OpenAI(api_key="YOUR_OPENAI_KEY")
# MODEL = "gpt-4o-mini"  # hoặc model bạn muốn

# Hàm trích text từ PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Hàm gửi prompt + parse JSON
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
        "Return ONLY the JSON object for the extracted data. Do not include any explanatory text or markdown code blocks (e.g., ```json...```).\n\n"
        f"TEXT:\n{text[:8000]}"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    content = response.text
    # 🌟 BƯỚC SỬA LỖI: Trích xuất chuỗi JSON khỏi khối mã Markdown
    # Tìm kiếm chuỗi nằm giữa ```json...``` hoặc ```...```
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    
    if match:
        json_string = match.group(1) # Lấy nội dung JSON đã được trích xuất
    else:
        json_string = content # Nếu không tìm thấy code block, sử dụng nội dung thô

    try:
        data = json.loads(json_string) # Thử parse JSON đã được trích xuất/làm sạch
        return data
    except Exception as e:
        print("⚠️ Không parse JSON được (Sau khi làm sạch):", e)
        # Nếu vẫn lỗi, lưu output thô để debug
        return {"raw_output": content}

# Hàm chính để xử lý cả folder PDF
def process_all_pdfs(pdf_folder="PDF-cases", output_folder="json-output"):
    os.makedirs(output_folder, exist_ok=True)
    pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    if not pdf_paths:
        print(f"❌ Không tìm thấy tệp PDF nào trong thư mục: {pdf_folder}. Vui lòng kiểm tra lại đường dẫn.")
        return

    for pdf in pdf_paths:
        print("📄 Processing:", pdf)
        try:
            text = extract_text_from_pdf(pdf)
            data = extract_case_info(text)

            # Lấy số thứ tự từ tên file PDF
            basename = os.path.basename(pdf)
            no_ext = os.path.splitext(basename)[0]
            case_num = no_ext.split("---")[0]
            json_filename = f"Case-{case_num}.json"

            out_path = os.path.join(output_folder, json_filename)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print("✅ Saved:", out_path)
        except Exception as e:
            print(f"🔥 Lỗi xử lý tệp {pdf}: {e}")

if __name__ == "__main__":
    process_all_pdfs()
