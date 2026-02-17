import os
import argparse
from PIL import Image
import torch

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


# -----------------------------
# Performance setup
# -----------------------------
torch.set_num_threads(os.cpu_count())

print("[*] Loading Surya models...")

# load backbone مرة واحدة فقط
foundation_predictor = FoundationPredictor()

# OCR model (يحتاج foundation)
recognition_predictor = RecognitionPredictor(
    foundation_predictor
)

# detector مستقل
detection_predictor = DetectionPredictor()

print("[*] Models loaded.")


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess(path):
    img = Image.open(path).convert("RGB")

    # Surya يعمل أفضل تحت 2048px عرض
    max_width = 1600
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize(
            (max_width, int(img.height * ratio)),
            Image.LANCZOS
        )

    return img


# -----------------------------
# OCR
# -----------------------------
@torch.inference_mode()
def run_ocr(image_path):

    image = preprocess(image_path)

    # Surya يقوم بالـ detection داخلياً هنا
    predictions = recognition_predictor(
        [image],
        det_predictor=detection_predictor
    )

    return predictions[0]


# -----------------------------
# Print result
# -----------------------------
def print_text(result):

    # result هو OCRResult object
    for line in result.text_lines:
        text = line.text.strip()
        if text:
            print(text)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("Image not found")
        return

    print("[*] Running OCR...")
    result = run_ocr(args.image)

    print("\n========== OCR RESULT ==========\n")
    print_text(result)


if __name__ == "__main__":
    main()
