"""
    Image 2 image image imanipulation methds
"""

from copy import copy

import numpy as np
import torch
from diffusers import FluxImg2ImgPipeline
from PIL.Image import Image

from arc_phash.image_manipulation.deepfake.utils import (
    ANIMALS,
    KEYWORDS,
    generate_random_prompt,
)
from arc_phash.image_manipulation.utils import resize_and_crop


class Img2Img:
    """img2img class"""

    def __init__(
        self,
        pipeline_name: str = "black-forest-labs/FLUX.1-dev",
        dtype: type = torch.bfloat16,
        device: str = "cpu",
        seed: int = 37,
        resize_size: int = 1024,
        prompt: str | None = None,
        save_path: str = "results",
        img_name: str = "",
        **kwargs,
    ):
        # pipeline
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            pipeline_name, torch_dtype=dtype
        )
        if device == "mps":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = device
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
            "guidance_scale": kwargs.get("guidance_scale", 2 + self.rng.uniform(0, 5)),
            "num_inference_steps": kwargs.get(
                "num_inference_steps", self.rng.integers(10, 25)
            ),
            "max_sequence_length": kwargs.get("max_sequence_length", 256),
            "strength": kwargs.get("strength", self.rng.uniform(0.2, 0.85)),
            "width": kwargs.get("width", self.resize_size),
            "height": kwargs.get("height", self.resize_size),
            "generator": torch.Generator(device=self.device),
        }

        # save path
        self.save_path = save_path
        self.img_name = img_name

    def resize_image(self, image: Image) -> Image:
        """Resize and crop the image for inpainting

        Args:
            image: input image

        Returns:
            resized and cropped image
        """
        return resize_and_crop(image, target_size=self.resize_size)

    def _generate_img2img(self, image: Image):
        """Generate img2img generative fill, append to results

        Args:
            image: image to inpaint
        """
        image = self.resize_image(image)
        pars = copy(self.pipe_kwargs)
        pars["image"] = image
        img2img_img = self.pipe(**pars).images[0]
        self.results.append(
            {
                "original": image,
                "img2img": img2img_img,
            }
        )

    def generate_img2img(self, image: Image | list[Image]):
        """Generate img2img on either a single input, or on a list of inputs, or
        single image

        Args:
            image: Either image or list of images for img2img manipulation.
            mask: Mask for inpainting for single image, will gen. Defaults to None.
        """
        if not isinstance(image, list):
            self._generate_img2img(image=image)
        else:
            for img in image:
                self._generate_img2img(image=img["image"])

    def get_results(self):
        """View the results"""
        return self.results

    def save_results(self):
        """Save the results"""
        for result in self.results:
            rand_idx = self.rng.integers(0, 9000)
            save_dest = f"{self.save_path}/{self.img_name}_img2img_{rand_idx}.png"
            result["img2img"].save(save_dest)
