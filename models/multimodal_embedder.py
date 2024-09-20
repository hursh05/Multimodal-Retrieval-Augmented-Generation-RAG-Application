from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from io import BytesIO

class MultimodalEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed(self, text, image_bytes):
        text_embedding = self.embed_text(text)
        image_embedding = self.embed_image(image_bytes)
        return text_embedding, image_embedding

    def embed_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        text_embeddings = self.model.get_text_features(**inputs)
        print("Text Embeddings:", text_embeddings)  # Debug print
        return text_embeddings

    def embed_image(self, image_bytes):
        if image_bytes is None:
            raise ValueError("No image bytes provided")

        try:
            image = Image.open(BytesIO(image_bytes))  # Wrap image_bytes in BytesIO
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")

        inputs = self.processor(images=image, return_tensors="pt")
        image_embeddings = self.model.get_image_features(**inputs)
        print("Image Embeddings:", image_embeddings)  # Debug print
        return image_embeddings
