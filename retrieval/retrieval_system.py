from pinecone import Pinecone, Index, ServerlessSpec

class RetrievalSystem:
    def __init__(self, api_key, index_name, host):
        self.pinecone = Pinecone(api_key=api_key)
        
        if index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index(
                name=index_name,
                dimension=1512,  
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        

        self.index = Index(index_name, host=host)

    def query(self, query_vector, top_k=10):
        return self.index.query(vector=query_vector, top_k=top_k)
