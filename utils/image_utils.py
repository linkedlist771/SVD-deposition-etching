import cv2
import numpy as np


def remove_watermark(watermark_image: np.array) -> np.array:
    """
    Remove the watermark by keeping only the bottom part of the image

    Args:
        watermark_image (np.array): Input image with watermark

    Returns:
        np.array: Image without the top watermark portion
    """
    # Get image dimensions - OpenCV image shape is (width, height, channels)
    width, height = watermark_image.shape[:2]

    # Define watermark region height (top 100 pixels)
    watermark_height = 100

    # Keep only the portion below the watermark
    result = watermark_image[watermark_height:, :]

    return result
