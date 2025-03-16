import argparse
import os

import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# from PIL import Image
from tqdm import tqdm

generator = torch.Generator("mps").manual_seed(92)

prompts = {
    "people": (
        "Change the face of this person, so that it looks like a different person. "
        "It should be realistic and high quality."
    ),
    "children": (
        "Change the face of this child, so that it looks like a different child. "
        "It should be realistic and high quality."
    ),
    "animals": (
        "Change the face of this animal, so that it looks like a different animal. "
        "It should be realistic and high quality."
    ),
    "nature": (
        "Change this object so that it looks like a different object. "
        "It should be realistic and high quality."
    ),
}


def perform_inpainting(image_path, mask_path, pipeline, prompt=None):
    # load base and mask image
    init_image = load_image(image_path).convert("RGB")
    mask_image = load_image(mask_path).convert("L")

    # width, height = init_image.size
    # # for stabilityai diffusion model, the width and height should be multiples of 8
    # width = max(1024, width - (width % 8))
    # height = max(1024, height - (height % 8))

    return pipeline(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        generator=generator,
        # width=width,
        # height=height,
    ).images[0]


# Example usage
def main(args):
    model_name = args.model_name
    data = args.data

    image_dir = f"data/{data}/images/"
    mask_dir = f"data/{data}/masks/"
    inpainted_dir = f"data/{data}/{model_name.replace('/', '-')}/"

    pipeline = AutoPipelineForInpainting.from_pretrained(model_name)
    pipeline.to("mps")
    # pipeline.set_progress_bar_config(disable=True)

    os.makedirs(inpainted_dir, exist_ok=True)

    # Find the first image in the image directory
    image_files = [
        f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))
    ]
    if not image_files:
        error_str = f"No images found in {image_dir}"
        raise ValueError(error_str)

    for image_filename in tqdm(image_files):
        if image_filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            print(f"First image filename: {image_filename}")

            image_path = os.path.join(image_dir, image_filename)
            mask_path = os.path.join(mask_dir, image_filename)
            inpainted_path = os.path.join(inpainted_dir, image_filename)

            inpainted_image = perform_inpainting(
                image_path, mask_path, pipeline, prompt=prompts[data]
            )
            inpainted_image.save(inpainted_path)


if __name__ == "__main__":
    """
    Example usage:
    python perform_inpainting.py people "runwayml/stable-diffusion-inpainting

    model_names:
        -   "runwayml/stable-diffusion-inpainting"
        -   "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" )
    """
    parser = argparse.ArgumentParser(description="Draw masks on images.")
    parser.add_argument("data", type=str, help="The data type name.")
    parser.add_argument("model_name", type=str, help="The model to use for inpainting.")
    args = parser.parse_args()
    main(args)
