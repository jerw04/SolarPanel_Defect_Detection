from PIL import Image
import numpy as np
from training.utils import preprocess_image

def test_preprocess_shape():
    img = Image.new("RGB", (300, 300))
    processed = preprocess_image(img)
    assert processed.shape == (1, 224, 224, 3)

def test_preprocess_normalized():
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    processed = preprocess_image(img)
    assert processed.max() <= 1.0
