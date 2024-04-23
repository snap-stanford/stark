import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def extract_patch(image, box):
    x1, y1, x2, y2 = box
    return image.crop((x1, y1, x2, y2))