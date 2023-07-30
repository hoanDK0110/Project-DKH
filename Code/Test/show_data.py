import networkx as nx
import matplotlib.pyplot as plt

# Đường dẫn tới file GML
gml_file_path = "./data/PHY/brain.gml"

# Đọc dữ liệu từ file GML và tạo đồ thị
G = nx.read_gml(gml_file_path)

PHY_nodes = list(G.nodes())
#PHY_weights_node = [G.nodes[node]['weight'] for node in PHY_nodes]
PHY_array = nx.adjacency_matrix(G).toarray()

print("PHY_nodes", PHY_nodes)
#print("PHY_weights_node", PHY_weights_node)
print("PHY_array", PHY_array)
# Hiển thị đồ thị
pos = nx.spring_layout(G, seed=42)  # Vị trí các nút trong đồ thị
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10)  # Hiển thị đồ thị với nhãn các nút

# Hiển thị trọng số trên các cạnh
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# Hiển thị
plt.show()
