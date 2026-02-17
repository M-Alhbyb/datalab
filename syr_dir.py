import os
import argparse
from pathlib import Path
from PIL import Image
import torch

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


# -----------------------------
# Setup
# -----------------------------
torch.set_num_threads(os.cpu_count())

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


print("[*] Loading Surya models...")

foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(
    foundation_predictor
)
detection_predictor = DetectionPredictor()

print("[*] Models loaded.")


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess(path):
    img = Image.open(path).convert("RGB")

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
def run_ocr(image):
    result = recognition_predictor(
        [image],
        det_predictor=detection_predictor
    )
    return result[0]


# -----------------------------
# Extract text
# -----------------------------
def extract_text(result):
    lines = [
        line.text.strip()
        for line in result.text_lines
        if line.text.strip()
    ]
    return "\n".join(lines)


# -----------------------------
# Process directory
# -----------------------------
def process_directory(input_dir, output_dir):

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in input_path.iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    ]

    if not images:
        print("No images found.")
        return

    print(f"[*] Found {len(images)} images")

    for i, img_path in enumerate(images, 1):

        print(f"[{i}/{len(images)}] Processing: {img_path.name}")

        try:
            image = preprocess(img_path)
            result = run_ocr(image)
            text = extract_text(result)

            out_file = output_path / (img_path.stem + ".txt")

            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)

        except Exception as e:
            print(f"[!] Failed: {img_path.name} -> {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Images directory")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print("Invalid directory")
        return

    process_directory(args.directory, "outputs")


if __name__ == "__main__":
    main()

