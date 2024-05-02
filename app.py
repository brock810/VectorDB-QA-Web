import os
from flask import Flask, render_template, request
from retrieval_component import retrieve_information
from transformers import BertTokenizer, BertLMHeadModel
import faiss
import numpy as np
import networkx as nx

os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

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

knowledge_graph = nx.Graph()
knowledge_graph.add_node('Paris', type='city')
knowledge_graph.add_node('France', type='country')
knowledge_graph.add_edge('Paris', 'France', relation='capital')

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertLMHeadModel.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    question = request.form['question']
    
    relevant_entities, relevant_relationships = retrieve_information(question, vector_db.index, knowledge_graph)
    
    retrieved_info = f'Relevant entities: {relevant_entities}\nRelevant relationships: {relevant_relationships}'
    
    generated_answer_result = generate_answer_with_retrieval(retrieved_info)
    
    return render_template('answer.html', answer=generated_answer_result)

def generate_answer_with_retrieval(query):
    relevant_entities, relevant_relationships = retrieve_information(query, vector_db.index, knowledge_graph)
    if relevant_entities:
        relevant_entities_str = f'Relevant entities: {relevant_entities}\n'
    else:
        relevant_entities_str = ''

    relevant_relationships_str = f'Relevant relationships: {relevant_relationships}\n'
    retrieved_info = relevant_entities_str + relevant_relationships_str

    input_tokens = tokenizer.encode(retrieved_info, add_special_tokens=True, return_tensors='pt')
    
    output_tokens = model.generate(input_tokens)
    generated_answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    return generated_answer


if __name__ == '__main__':
    app.run(debug=True, port=8084)
