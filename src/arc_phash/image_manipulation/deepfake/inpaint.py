"""
    Inpainting generation methods
"""

from copy import copy

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL.Image import Image

from arc_phash.image_manipulation.deepfake.utils import (
    ANIMALS,
    KEYWORDS,
    generate_random_prompt,
)
from arc_phash.image_manipulation.utils import resize_and_crop


class InPainter:
    """Inpainting class"""

    def __init__(
        self,
        pipeline_name: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device: str = "cpu",
        seed: int = 37,
        resize_size: int = 1024,
        prompt: str | None = None,
        save_path: str = "results",
        img_name: str = "",
        **kwargs,
    ):
        # pipeline
        self.pipe = AutoPipelineForInpainting.from_pretrained(pipeline_name)
        if device == "mps":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.pipe.to(device)

        # image editing
        self.rng = np.random.default_rng(seed=seed)
        self.resize_size = resize_size

        # results output
        self.results = []

        # pipeline kwargs
        prompt = (
            prompt
            if prompt is not None
            else generate_random_prompt(keywords=KEYWORDS, animals=ANIMALS, seed=seed)
        )
        self.pipe_kwargs = {
            "prompt": prompt,
            "device": self.device,
            "guidance_scale": kwargs.get("guidance_scale", 4.5),
            "strength": kwargs.get("strength", self.rng.uniform(0.5, 0.89)),
            "num_inference_steps": kwargs.get("num_inference_steps", 100),
            "safety_checker": kwargs.get("safety_checker"),
            "generator": torch.Generator(device=device),
        }

        # save path
        self.save_path = save_path
        self.img_name = img_name

    def generate_random_mask(self, image: Image) -> Image:
        """Generate a random mask for for an image.

        Args:
            image: input image on which to create random mask

        Returns:
            random mask of same dimensions
        """
        mask = image.copy()
        mask = mask.convert("L")
        mask = mask.point(lambda x: 0)
        blob_size = self.rng.integers(
            self.resize_size // 10, self.resize_size - self.resize_size // 10
        )
        x, y = self.rng.integers(0, self.resize_size - blob_size, 2)
        for i in range(blob_size):
            for j in range(blob_size):
                mask.putpixel(x + i, y + j, 255)

        return self.pipe.mask_processor.blur(mask, blur_factor=12)

    def resize_image(self, image: Image) -> Image:
        """Resize and crop the image for inpainting

        Args:
            image: input image

        Returns:
            resized and cropped image
        """
        return resize_and_crop(image, target_size=self.resize_size)

    def _generate_inpainting(self, image: Image, mask: Image = None):
        """Generate inpainting based on a mask, with a randomly generated mask if none
        is given

        Args:
            image: image to inpaint
            mask: mask for inpainting. Defaults to None, will randomly generate a mask
        """
        image = self.resize_image(image)
        mask = mask if mask is not None else self.generate_random_mask(image)
        kwargs = copy(self.pipe_kwargs)
        kwargs["image"] = image
        kwargs["mask"] = mask
        inpaint_image = self.pipe(**kwargs)
        self.results.append(
            {"original": image, "inpainted": inpaint_image, "mask": mask}
        )

    def generate_inpainting(
        self, image: Image | list[dict[str, Image]], mask: Image = None
    ):
        """Generate inpainting on either a single input, or on a list of inputs with
        dictionary structure:
                {'image': image, 'mask': mask | None}

        Args:
            image: Either image or list of image or (image, mask) pairs for inpainting.
            mask: Mask for inpainting for single image, will gen. Defaults to None.
        """
        if not isinstance(image, list):
            self._generate_inpainting(image=image, mask=mask)
        else:
            for img in image:
                self._generate_inpainting(image=img["image"], mask=img.get("mask"))

    def get_results(self):
        """View the results"""
        return self.results

    def save_results(self):
        """Save the results"""
        for result in self.results:
            rand_idx = self.rng.integers(0, 9000)
            save_dest = f"{self.save_path}/{self.img_name}_inpaint_{rand_idx}.png"
            result["inpainted"].save(save_dest)
