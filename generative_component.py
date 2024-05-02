from transformers import BertTokenizer, BertLMHeadModel
from retrieval_component import retrieve_information

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertLMHeadModel.from_pretrained(model_name)

def generate_answer_with_retrieval(query):
    relevant_entities, relevant_relationships = retrieve_information(query)
    retrieved_info = f'Relevant entities: {relevant_entities}\nRelevant relationships: {relevant_relationships}'
    input_tokens = tokenizer.encode(retrieved_info, add_special_tokens=True, return_tensors='pt')
    
    output_tokens = model.generate(input_tokens)
    generated_answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    return generated_answer
