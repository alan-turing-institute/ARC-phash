import argparse
import os

import cv2
import numpy as np


def draw_mask(image_dir, save_dir, image_name):
    # Load the image
    image = cv2.imread(image_dir + image_name)
    if image is None:
        error_str = f"Image not found at {image_dir + image_name}"
        raise ValueError(error_str)

    # Create a blank mask
    mask = np.zeros_like(image)

    # Determine cursor size based on image size
    cursor_size = min(image.shape[0], image.shape[1]) // 15

    # Function to handle mouse events
    def draw(event, x, y, flags, param):
        _ = param  # Ignore the unused parameter warning
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(mask, (x, y), cursor_size, (255, 255, 255), -1)

    # Create a window and set the mouse callback
    cv2.namedWindow("Draw Mask")
    cv2.setMouseCallback("Draw Mask", draw)

    while True:
        # Display the image and mask
        combined = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        cv2.imshow("Draw Mask", combined)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close the window
    cv2.destroyAllWindows()

    # Save the mask
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mask_path = save_dir + image_name
    cv2.imwrite(mask_path, mask)
    print(f"Mask saved at {mask_path}")


def main(args):
    data = args.data
    image_dir = f"../data/{data}/images/"
    image_files = [
        f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))
    ]

    if not image_files:
        error_str = f"No images found in {image_dir}"
        raise ValueError(error_str)
    for image_path in image_files:
        save_dir = f"../data/{data}/masks/"

        draw_mask(image_dir, save_dir, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw masks on images.")
    parser.add_argument("data", type=str, help="The data type name.")
    args = parser.parse_args()
    main(args)
