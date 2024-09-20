from transformers import pipeline

class LLMIntegration:
    def __init__(self):
        self.llm = pipeline("text-generation", model="gpt-2")

    def generate_response(self, text, images):
        input_text = f"{text} {images}"  # Modify as needed
        return self.llm(input_text, max_length=150)[0]["generated_text"]
