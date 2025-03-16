 Data Generation

This folder contains scripts for generating and manipulating data for the ARC-phash project. Below is a description of each script and its usage.

## Scripts

There are three scripts for data generation, `draw_masks.py`, `flip_images.py`, and `perform_inpainting.py`. They should be run in that order unless you wish to have mirrored images with different masks, in which case you should generate the masks first.

You code expects a directory of images placed in the parent folder of the codebase, structured as such `/data/[image_subject]/images/`. Currently the scripts expect `image_subject` to be `people`, `animals`, or `nature`.

### 1. `draw_masks.py`

This script allows you to draw masks on images manually. The masks can be used for inpainting or other image processing tasks.

#### Usage

```bash
python draw_masks.py <data>
```

#### Arguments

- `<data>`: The name of the data type (e.g., `people`, `animals`, `nature`).

#### Example

```bash
python draw_masks.py people
```

### 2. `flip_images.py`

This script flips images and their corresponding masks horizontally. It can be used to augment the dataset by creating mirrored versions of the images.

#### Usage

No arguments are required. The script processes images and masks in predefined directories.

#### Example

```bash
python flip_images.py
```

### 3. `perform_inpainting.py`

This script performs inpainting on images using a specified model. Inpainting is the process of reconstructing lost or deteriorated parts of images.

#### Usage

```bash
python perform_inpainting.py <data> <model_name>
```

#### Arguments

- `<data>`: The name of the data type (e.g., `people`, `animals`, `nature`).
- `<model_name>`: The name of the model to use for inpainting (e.g., `runwayml/stable-diffusion-inpainting`).

#### Example

```bash
python perform_inpainting.py people "runwayml/stable-diffusion-inpainting"
```

## Notes

- Ensure that the required dependencies are installed before running the scripts.
- The scripts assume a specific directory structure for the input data. Make sure your data is organized accordingly.
