"""
    Utility functions for deep fake image manipulation
"""


def generate_prompt(test=False):
    """Generate a random prompt for use in AI image manipulation.

    Args:
        test: Optionally return test string. Defaults to False.
    """
    if test:
        return "A cat holding a sign that says hello world"

    return NotImplementedError
