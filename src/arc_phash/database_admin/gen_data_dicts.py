"""
    Given the correct metadat, generate the database dictionaries ready for insertion
    into milvus
"""

from copy import copy
from typing import Any

import ulid
from PIL import Image

from arc_phash.hashing import dhash, pdqhash


def produce_data_dicts(img: Image, **kwargs) -> dict[str, dict[str, Any]]:
    """for a single image produce a database row vect

    Args:
        img: input image
        kwargs:
            {
                'ai': flag for ai,
                'file_path': file path
            }

    Returns:
        database row dicts for each hash
    """
    unique_id = str(ulid.new())
    data_dict = {
        "ai": kwargs["ai"],
        "ulid": unique_id,
        "file_path": kwargs["file_path"],
    }
    pdq_dict = copy(data_dict)
    pdq_dict.update({"vector": pdqhash.pdqhash_img(img)})
    dhash_dict = copy(data_dict)
    dhash_dict.update({"vector": dhash.dhash_img(img)})
    return {
        "pdqhash": pdq_dict,
        "dhash": dhash_dict,
    }
