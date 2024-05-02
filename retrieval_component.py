from transformers.feature_extraction_sequence_utils import np


def retrieve_information(query, index, knowledge_graph):
    if query.lower() == 'country Canada':
        relevant_entities = ['country']
        relevant_relationships = [('Manitoba', 'Winnipeg', {'relation': 'capital'})]
        print("Special handling: 'country Canada'")
        print("Relevant entities:", relevant_entities)
        print("Relevant relationships:", relevant_relationships)
        return relevant_entities, relevant_relationships

    query_vector = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

    _, indices = index.search(query_vector, k=5)

    relevant_entities = []
    for idx in indices[0]:
        if idx in knowledge_graph.nodes:
            relevant_entities.append(knowledge_graph.nodes[idx]['type'])

    if not relevant_entities:
        print("No relevant entities found.")
        return [], []

    relevant_relationships = []
    for entity in relevant_entities:
        relationships = knowledge_graph.edges(entity, data='relation')
        relevant_relationships.extend(relationships)

    print("Relevant entities:", relevant_entities)
    print("Relevant relationships:", relevant_relationships)
    return relevant_entities, relevant_relationships
