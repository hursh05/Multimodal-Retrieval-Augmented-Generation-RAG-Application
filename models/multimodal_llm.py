

class MultimodalLLM:
    def __init__(self, embedder, retrieval_system):
        self.embedder = embedder
        self.retrieval_system = retrieval_system

    def process_query(self, query_text, query_image_bytes=None):
        
        query_embedding_text = self.embedder.embed_text(query_text)

        
        query_embedding_image = None
        if query_image_bytes is not None:
            query_embedding_image = self.embedder.embed_image(query_image_bytes)


        if query_embedding_image is not None:
            query_embedding = query_embedding_text + query_embedding_image
        else:
            query_embedding = query_embedding_text

        if query_embedding is None:
            raise ValueError("Query embedding is None. Check the input data and embedding generation.")

        query_vector = query_embedding.detach().numpy().tolist()  

        results = self.retrieval_system.query(query_vector)
        return results
