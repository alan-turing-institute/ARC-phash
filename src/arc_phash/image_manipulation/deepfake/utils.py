"""
    Utility functions for deep fake image manipulation
"""

import numpy as np
from wonderwords import RandomSentence

KEYWORDS = ["roam", "grungy", "journal", "drive", "ridge"]
ANIMALS = [
    "man",
    "woman",
    "cat",
    "dog",
    "bird",
    "fish",
    "elephant",
    "giraffe",
    "lion",
    "tiger",
    "bear",
    "wolf",
    "fox",
    "rabbit",
    "deer",
    "moose",
    "squirrel",
    "chipmunk",
    "beaver",
    "otter",
    "raccoon",
    "skunk",
]


def generate_random_prompt(
    keywords: list[str],
    animals: list[str],
    test: bool = False,
    seed: int = 37,
) -> str:
    """Generate a random prompt for use in AI image manipulation.

    Args:
        test: Optionally return test string. Defaults to False.
        seed: Seed for randomness. Defaults to 37.
        keywords: list of keywords for prompt generation.
        animals: list of animals for prompt generation.

    Returns:
        prompt, str
    """
    if test:
        return "A cat holding a sign that says hello world"

    rng = np.random.default_rng(seed=seed)
    sentence_tool = RandomSentence()

    keyword = rng.choice(keywords)
    animal = rng.choice(animals)
    return f"{keyword} {animal} {sentence_tool.bare_bone_with_adjective()}"
