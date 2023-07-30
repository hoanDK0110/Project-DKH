import os
import networkx as nx
import xml.etree.ElementTree as ET

def xml_to_gml(input_file, output_file):
    # Phân tích dữ liệu XML
    tree = ET.parse(input_file)
    print(tree)
    root = tree.getroot()
    # Namespace của XML


    # Tạo đồ thị NetworkX
    G = nx.DiGraph()

    # Thêm thông tin node vào đồ thị
    nodes = root.find(".//nodes")
    for node in nodes:
        node_id = node.get("id")
        coordinates = node.find("../coordinates")  
        x = float(coordinates.find("../x").text)
        y = float(coordinates.find("../y").text)
        G.add_node(node_id, x=x, y=y)

    # Thêm thông tin cạnh vào đồ thị
    links = root.find("networkStructure/links")
    for link in links.findall("link"):
        link_id = link.get("id")
        source = link.find("source").text
        target = link.find("target").text
        additionalModules = link.find("additionalModules")  
        capacity = float(additionalModules.find("capacity").text)
        cost = float(additionalModules.find("cost").text)
        G.add_edge(source, target, capacity=capacity, cost=cost)


    # Xử lý thông tin nhu cầu (demands) và thêm vào các cạnh
    demands = root.find("demands")
    for demand in demands.findall('demand'):
        source = demand.find('source').text
        target = demand.find('target').text
        demandValue = float(demand.find('demandValue').text)
        # Tìm cạnh tương ứng trong đồ thị
        edge = G.get_edge_data(source, target)

        # Nếu cạnh tồn tại, thêm thông tin nhu cầu vào
        if edge is not None:
            edge_id = edge["id"]
            G.edges[source, target, edge_id]["demand"] = demandValue

    # Lưu đồ thị dưới định dạng GML vào file
    nx.write_gml(G, output_file)


def xml_files_to_gmls(input_folder):
    # Tạo thư mục đầu ra nếu nó chưa tồn tại
    if not os.path.exists('data'):
        os.makedirs('data')

    # Liệt kê tất cả các file trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        # Kiểm tra nếu file có phần mở rộng là ".xml"
        if filename.endswith(".xml"):
            xml_file_path = os.path.join(input_folder, filename)

            # Đọc nội dung từ file XML
            with open(xml_file_path, 'r') as xml_file:
                xml_data = xml_file.read()

            # Chuyển đổi và lưu file GML
            xml_to_gml(xml_data, xml_file_path)

# Đường dẫn tới thư mục chứa các file XML
input_file = './sndlib-networks-xml/abilene.xml'
output_file = './data/abilene.gml'
xml_to_gml(input_file, output_file)


