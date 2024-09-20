import torch
from transformers import CLIPProcessor, CLIPModel

class MultimodalEmbedding:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def create_embeddings(self, text, images):
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return embeddings
