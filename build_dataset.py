import os, json, glob

JSON_FOLDER = "json-output"
IMAGE_FOLDER = "image-output"
OUTPUT_FILE = "dataset_ready.jsonl"

records = []

for json_path in glob.glob(os.path.join(JSON_FOLDER, "*.json")):
    case_id = os.path.splitext(os.path.basename(json_path))[0]  # ví dụ: Case-1
    case_img_folder = os.path.join(IMAGE_FOLDER, case_id)
    image_paths = []
    if os.path.exists(case_img_folder):
        for img_file in os.listdir(case_img_folder):
            image_paths.append(os.path.join(case_img_folder, img_file))

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Gộp text thành một đoạn duy nhất
    text_input = (
        f"Patient information: {data.get('patient_information', 'Not mentioned')}. "
        f"Chief complaint: {data.get('chief_complaint', 'Not mentioned')}. "
        f"History of present illness: {data.get('history_of_present_illness', 'Not mentioned')}. "
        f"Exposure and epidemiology: {data.get('exposure_and_epidemiology', 'Not mentioned')}. "
        f"Vitals: {data.get('vitals', 'Not mentioned')}. "
        f"Physical exam: {data.get('physical_exam', 'Not mentioned')}. "
        f"Labs and diagnostics: {data.get('labs_and_diagnostics', 'Not mentioned')}. "
        f"Management and clinical course: {data.get('management_and_clinical_course', 'Not mentioned')}."
    )

    record = {
        "id": case_id,
        "text_input": text_input,
        "label": data.get("disease_name_short", "Unknown"),
        "images": image_paths
    }
    records.append(record)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for r in records:
        json.dump(r, out, ensure_ascii=False)
        out.write("\n")

print(f"✅ Đã tạo {OUTPUT_FILE} với {len(records)} cases.")
