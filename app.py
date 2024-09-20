import streamlit as st
from retrieval.retrieval_system import RetrievalSystem
from models.multimodal_llm import MultimodalLLM
from extract.pdf_extractor import PDFExtractor
from models.multimodal_embedder import MultimodalEmbedder
from pinecone import Pinecone, ServerlessSpec
from io import BytesIO

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_HOST = st.secrets["PINECONE_HOST"]
INDEX_NAME = st.secrets["INDEX_NAME"]


pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_HOST)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME, 
        dimension=1536, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  
        )
    )

def main():
    st.title("Multimodal Retrieval-Augmented Generation")


    try:
        retrieval_system = RetrievalSystem(
            api_key=PINECONE_API_KEY, 
            index_name=INDEX_NAME, 
            host=PINECONE_HOST
        )
    except Exception as e:
        st.error(f"Failed to initialize the retrieval system: {e}")
        return

    pdf_file = st.file_uploader("Upload a PDF", type="pdf")

    if pdf_file is not None:
        pdf_extractor = PDFExtractor(pdf_file)
        text, images, structured_data = pdf_extractor.extract_data()


        st.write("Extracted Text:", text)
        st.image(images, caption="Extracted Images", use_column_width=True)

        embedder = MultimodalEmbedder()
        multimodal_llm = MultimodalLLM(embedder, retrieval_system)


        query_text = st.text_input("Enter your query text:")


        query_image = st.file_uploader("Upload an image for the query (optional):", type=["png", "jpg", "jpeg"])

        if query_image is not None:
            query_image_bytes = query_image.read() 
        else:
            query_image_bytes = None

        if query_text:
            if query_image_bytes:
                print(query_image_bytes[:10])  

            try:
                results = multimodal_llm.process_query(query_text, query_image_bytes)
                st.write("Results:", results)
            except ValueError as e:
                st.error(f"An error occurred while processing your query: {e}")

if __name__ == "__main__":
    main()
