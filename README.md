# Multimodal-Retrieval-Augmented-Generation-RAG-Application


Overview
This project implements a Multimodal Retrieval-Augmented Generation (RAG) system capable of processing both text and image inputs for query-based content retrieval and answer generation. The pipeline combines PDF text and image extraction, multimodal embeddings using CLIP, and query-based retrieval from Pinecone. The system generates answers by utilizing a multimodal large language model (LLM) based on both retrieved text and image data.

Workflow Diagram
You can visualize the entire workflow with this image:
![Screenshot 2024-09-19 205005](https://github.com/user-attachments/assets/70368b2c-7c65-4be5-85a1-8c457451655d)

Features
Multimodal Input Processing: Handles both text and image inputs without OCR for images.
Embedding in Shared Vector Space: Text and image content are embedded into a shared multimodal vector space using CLIP.
Semantic Retrieval: User queries are processed to fetch relevant content (text and image) from the Pinecone vector store.
Answer Generation: A multimodal LLM generates answers based on the retrieved content.
Interactive Interface: The application provides a Streamlit-based UI where users can upload PDFs, submit queries, and receive responses.


The workflow image has been successfully saved as /mnt/data/workflow_image.png. Below is the requirements.txt and a detailed README for your GitHub repository.

requirements.txt
plaintext
Copy code
torch==2.0.0
transformers==4.30.0
Pillow==9.3.0
streamlit==1.24.0
pinecone-client==2.2.1
PyMuPDF==1.22.1
README.md
Multimodal Retrieval-Augmented Generation (RAG) Application
Overview
This project implements a Multimodal Retrieval-Augmented Generation (RAG) system capable of processing both text and image inputs for query-based content retrieval and answer generation. The pipeline combines PDF text and image extraction, multimodal embeddings using CLIP, and query-based retrieval from Pinecone. The system generates answers by utilizing a multimodal large language model (LLM) based on both retrieved text and image data.

Workflow Diagram
You can visualize the entire workflow with this image:

Features
Multimodal Input Processing: Handles both text and image inputs without OCR for images.
Embedding in Shared Vector Space: Text and image content are embedded into a shared multimodal vector space using CLIP.
Semantic Retrieval: User queries are processed to fetch relevant content (text and image) from the Pinecone vector store.
Answer Generation: A multimodal LLM generates answers based on the retrieved content.
Interactive Interface: The application provides a Streamlit-based UI where users can upload PDFs, submit queries, and receive responses.
Prerequisites
Make sure you have the following tools and libraries installed:

Python 3.8+
PIP (Python package manager)
Pinecone account for vector storage

Installation
Clone the repository:

git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag

Install dependencies: Use the following command to install all required libraries:

pip install -r requirements.txt
Pinecone Setup: Create a free Pinecone account and get the API key and index details.

Configure API Keys: In .streamlit/secrets.toml, add your Pinecone API details:

PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_HOST = "your-pinecone-host"
INDEX_NAME = "your-index-name"

Usage
Run the Application: You can start the application using Streamlit:

streamlit run app.py
Upload PDF: Use the Streamlit interface to upload a PDF file. The system will extract text and images from the document.

Submit Query: Enter a text query and optionally upload an image for multimodal query generation.

View Results: The application will display the relevant text and images, followed by a generated answer.


File Structure

app.py: Main application file that integrates the UI and backend logic.
extract/: Contains scripts for extracting content from PDFs.
pdf_extract.py: Extracts text and images from PDF files.
image_extractor.py: Handles image extraction.
models/: Contains models for generating multimodal embeddings and integrating LLM.
multimodal_embedder.py: Embeds text and images into a shared vector space using CLIP.
multimodal_llm.py: Processes user queries and retrieves relevant documents.
generation/: Handles LLM generation for multimodal inputs.
llm_generation.py: Generates responses based on retrieved text and images.
retrieval/: Contains scripts for handling semantic search using Pinecone.
retrieval_system.py: Interacts with Pinecone for query-based retrieval.


Future Work
Fine-tuning Models: Further fine-tune the multimodal LLM for improved responses.
Performance Optimization: Explore ways to optimize semantic retrieval for larger datasets.
Additional Features: Support other file formats for input, such as Word documents or images.

Contributors

Hursh Karnik - BTech CSE (AI/ML)
