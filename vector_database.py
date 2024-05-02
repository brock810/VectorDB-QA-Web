import faiss
import numpy as np

class VectorDatabase:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension) 
    
    def add_vectors(self, vectors):
        self.index.add(np.array(vectors))
    
    def search_vectors(self, query_vector, k=5):
        _, indices = self.index.search(query_vector, k)
        return indices
        
vector_db = VectorDatabase(dimension=10)
