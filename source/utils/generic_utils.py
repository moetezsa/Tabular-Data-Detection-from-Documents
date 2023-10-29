import io
import numpy as np
from PIL import Image
import cv2 as cv


def image_input(image_content: bytes):
    image_from_memory = np.asarray(bytearray(io.BytesIO(image_content).read()), dtype="uint8")
    decoded_image_bgr = cv.imdecode(image_from_memory, cv.IMREAD_COLOR)
    decoded_image_rgb = cv.cvtColor(decoded_image_bgr, cv.COLOR_BGR2RGB)
    return decoded_image_rgb


def get_bytestream_from_image(image: np.ndarray) -> io.BytesIO:
    """
    Return an image bytestream from an image array
    :param image: image array
    :return: byte stream of image
    """
    preprocessed_image = Image.fromarray(image)
    return_image = io.BytesIO()
    preprocessed_image.save(return_image, "PNG")
    return_image.seek(0)
    return return_image
