import argparse
import os

from PIL import Image

from arc_phash.ai_detection.detector import AIDetector


def main(model_str: AIDetector, image_dir: str):
    detector = AIDetector(model_str)
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            img = Image.open(image_path)
            result = detector(img)
            print(f"Detection result for {filename}: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI detection on images.")
    parser.add_argument(
        "model_str", type=str, help="The model string for the AI detector."
    )
    parser.add_argument(
        "image_directory",
        type=str,
        help="The directory containing images to be processed.",
    )
    args = parser.parse_args()

    main(args.model_str, args.image_directory)
