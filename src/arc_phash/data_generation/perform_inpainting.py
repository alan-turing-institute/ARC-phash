import argparse
import os

import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
from torchvision import transforms

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    device="mps",
)

generator = torch.Generator("mps").manual_seed(92)


def perform_inpainting(
    image_path, mask_path
):  # , model_name="facebook/dpt-inpainting"):
    # Load the image and mask
    # image = Image.open(image_path).convert("RGB")
    # mask = Image.open(mask_path).convert("L")

    # load base and mask image
    init_image = load_image(image_path)
    mask_image = load_image(mask_path)
    # Convert PIL images to PyTorch tensors and move to MPS device
    transform = transforms.ToTensor()
    init_image = transform(init_image).unsqueeze(0).to("mps")
    mask_image = transform(mask_image).unsqueeze(0).to("mps")

    prompt = "make random changes to this image"
    image = pipeline(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        generator=generator,
    ).images[0]
    make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    # Save the inpainted image
    return image
    # # Load the model and processor
    # model = DPTForImageInpainting.from_pretrained(model_name)
    # processor = DPTImageProcessor.from_pretrained(model_name)

    # # Preprocess the image and mask
    # inputs = processor(images=image, masks=mask, return_tensors="pt")

    # # Perform inpainting
    # with torch.no_grad():
    #     outputs = model(**inputs)

    # # Get the inpainted image
    # inpainted_image = processor.post_process(outputs.logits, inputs["pixel_values"])

    # # Convert tensor to PIL image
    # return Image.fromarray(inpainted_image.squeeze().permute(1, 2, 0).byte().numpy())


# Example usage
def main(args):
    data = args.data
    image_dir = f"../data/{data}/images/"
    mask_dir = f"../data/{data}/masks/"
    inpainted_dir = f"../data/{data}/inpainted/"
    os.makedirs(inpainted_dir, exist_ok=True)

    # Find the first image in the image directory
    image_files = [
        f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))
    ]
    if not image_files:
        error_str = f"No images found in {image_dir}"
        raise ValueError(error_str)

    image_filename = image_files[0]
    print(f"First image filename: {image_filename}")

    image_path = os.path.join(image_dir, image_filename)
    mask_path = os.path.join(mask_dir, image_filename)
    inpainted_path = os.path.join(inpainted_dir, image_filename)

    inpainted_image = perform_inpainting(image_path, mask_path)
    inpainted_image.save(inpainted_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw masks on images.")
    parser.add_argument("data", type=str, help="The data type name.")
    args = parser.parse_args()
    main(args)
