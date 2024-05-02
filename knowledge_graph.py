import networkx as nx

knowledge_graph = nx.Graph()
knowledge_graph.add_node('Paris', type='city')
knowledge_graph.add_node('France', type='country')
knowledge_graph.add_edge('Paris', 'France', relation ='capital')
