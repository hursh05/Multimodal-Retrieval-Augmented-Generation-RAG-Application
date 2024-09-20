
import fitz  

class PDFExtractor:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def extract_data(self):
        document = fitz.open(stream=self.pdf_file.read(), filetype="pdf")
        text = ""
        images = []
        structured_data = []

        for page in document:
            text += page.get_text()
            images_info = page.get_images(full=True)

            for img_index, img in enumerate(images_info):
                xref = img[0]
                try:
                    base_image = document.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append(image_bytes)  # Append raw image bytes
                except ValueError as e:
                    print(f"Error extracting image with xref {xref}: {e}")

        document.close()
        return text, images, structured_data
