from PIL import Image
import os

class ImageExtractor:
    def __init__(self, image_path):
        self.image_path = image_path

    def extract_image(self):
        return Image.open(self.image_path)
