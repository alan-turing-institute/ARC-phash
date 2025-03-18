import os
import random

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

for data in ["people", "animals", "nature"]:
    # Define the directory containing the images
    input_dir = f"data/{data}/images/"
    flipped_dir = f"data/{data}/flipped/"
    cropped_dir = f"data/{data}/cropped/"

    # Create the output directory if it doesn't exist
    os.makedirs(flipped_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    # Define the transformation to flip the image horizontally
    flip_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(
                p=1.0
            )  # Flip the image with 100% probability
        ]
    )

    # Define the transformation to apply a random crop with non-specific output size
    def random_crop(img):
        width, height = img.size
        crop_width = random.randint(int(0.5 * width), int(0.8 * width))
        crop_height = random.randint(int(0.5 * height), int(0.8 * height))

        if width < crop_width or height < crop_height:
            raise ValueError("Crop size must be smaller than the image size")

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        return img.crop((left, top, right, bottom))

    # Loop through all files in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            # Open the image file
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # Apply the transformation
            flipped_img = flip_transform(img)
            cropped_img = random_crop(img)

            # Save the flipped image to the output directory
            flipped_path = os.path.join(flipped_dir, filename)
            flipped_img.save(flipped_path)

            # Save the flipped image to the output directory
            cropped_path = os.path.join(cropped_dir, filename)
            cropped_img.save(cropped_path)

    print(f"{data} images have been flipped and saved successfully.")
