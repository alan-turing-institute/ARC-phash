import os

from PIL import Image
from torchvision import transforms

for data in ["animals"]:
    for context in ["images", "masks"]:
        # Define the directory containing the images
        input_dir = f"data/{data}/{context}/"
        output_dir = f"data/{data}/{context}/"

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define the transformation to flip the image horizontally
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(
                    p=1.0
                )  # Flip the image with 100% probability
            ]
        )

        # Loop through all files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                # Open the image file
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path)

                # Apply the transformation
                flipped_img = transform(img)

                # Save the flipped image to the output directory
                output_path = os.path.join(output_dir, f"flipped_{filename}")
                flipped_img.save(output_path)

    print(f"{data} images have been flipped and saved successfully.")
