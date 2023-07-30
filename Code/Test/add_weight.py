import random
import networkx as nx

def add_random_weights_to_nodes(graph):
    for node in graph.nodes:
        graph.nodes[node]['weight'] = random.randint(1000, 1500)

def add_random_weights_to_edges(graph):
    for edge in graph.edges:
        graph.edges[edge]['weight'] = random.randint(10, 15)

def export_to_gml_with_weights(graph, filename):
    nx.write_gml(graph, filename)

def process_gml_file_with_weights(input_file, output_file):
    # Đọc dữ liệu và tạo đồ thị từ file GML
    graph = nx.read_gml(input_file)

    # Thêm trọng số random vào các node
    add_random_weights_to_nodes(graph)

    # Thêm trọng số random vào các cạnh
    add_random_weights_to_edges(graph)

    # Xuất dữ liệu mới ra file GML
    export_to_gml_with_weights(graph, output_file)

if __name__ == "__main__":
    input_file = "./data/PHY/giul39.gml"
    output_file = "./data_import/data_PHY/giul39.gml"
    process_gml_file_with_weights(input_file, output_file)
