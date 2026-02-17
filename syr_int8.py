import os
from pathlib import Path
from PIL import Image
import torch
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# --- تسريع CPU ---
os.environ["TORCH_INFERENCE_MODE"] = "1"  # يقلل overhead أثناء inference
torch.set_num_threads(os.cpu_count())     # استخدام كل الأنوية

# --- إعداد المجلدات ---
INPUT_DIR = "samples"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- إعداد النماذج ---
foundation_predictor = FoundationPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor)
det_predictor = DetectionPredictor()

# --- دالة استخراج النص ---
def extract_text(image_path):
    image = Image.open(image_path).convert("L")  # تحويل grayscale
    results = rec_predictor([image], det_predictor=det_predictor)
    text_lines = [line.text for line in results[0].text_lines]
    return "\n".join(text_lines)

# --- معالجة كل الصور في المجلد ---
for image_file in Path(INPUT_DIR).glob("*.*"):
    try:
        text = extract_text(str(image_file))
        output_file = Path(OUTPUT_DIR) / f"{image_file.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[+] Done: {image_file.name} -> {output_file.name}")
    except Exception as e:
        print(f"[!] Error processing {image_file.name}: {e}")
