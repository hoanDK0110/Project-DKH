import os
import networkx as nx
import xml.etree.ElementTree as ET

def xml_to_gml(input_xml, output_gml):
    # Đọc dữ liệu từ chuỗi XML
    root = ET.fromstring(input_xml)

    # Namespace của XML
    namespace = {'ns': ''}

    # Khởi tạo đồ thị
    G = nx.Graph()

    # Xử lý các phần tử trong dữ liệu XML và thêm các nút vào đồ thị
    for node in root.findall('ns:networkStructure/ns:nodes/ns:node', namespace):
        node_id = node.get("id")
        coordinates = node.find("coordinates")  
        x = float(coordinates.find("x").text)
        y = float(coordinates.find(".y").text)
        G.add_node(node_id, x=x, y=y)

    # Xử lý các phần tử trong dữ liệu XML và thêm các cạnh vào đồ thị
    for link in root.findall('ns:networkStructure/ns:links/ns:link', namespace):
        source = link.find('ns:source', namespace).text
        target = link.find('ns:target', namespace).text
        capacity = float(link.find('ns:additionalModules/ns:addModule/ns:capacity', namespace).text)
        cost = float(link.find('ns:additionalModules/ns:addModule/ns:cost', namespace).text)
        G.add_edge(source, target, capacity=capacity, cost=cost)

    # Lưu đồ thị vào file GML với tên tương tự file XML
    base_name = os.path.basename(output_gml)
    gml_file = os.path.join('data', os.path.splitext(base_name)[0] + ".gml")
    nx.write_gml(G, gml_file)

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

# Thay đổi đường dẫn thư mục đầu vào và thư mục đầu ra tương ứng ở đây
input_folder = './sndlib-networks-xml/'

# Gọi hàm để chuyển đổi và lưu tất cả các file XML thành GML với tên tương tự
xml_files_to_gmls(input_folder)