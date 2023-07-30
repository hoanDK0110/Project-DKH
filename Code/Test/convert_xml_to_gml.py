import networkx as nx
import xml.etree.ElementTree as ET

def xml_to_gml(xml_file_path, gml_file_path):
    # Read the XML data from the file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Define the GML graph
    gml_graph = nx.Graph()

    # Extract nodes and their coordinates
    for node in root.findall('.//{http://sndlib.zib.de/network}node'):
        node_id = node.get('id')
        x = float(node.find('.//{http://sndlib.zib.de/network}x').text)
        y = float(node.find('.//{http://sndlib.zib.de/network}y').text)
        gml_graph.add_node(node_id, x=x, y=y)

    # Extract links and their additional modules
    for link in root.findall('.//{http://sndlib.zib.de/network}link'):
        link_id = link.get('id')
        source = link.find('.//{http://sndlib.zib.de/network}source').text
        target = link.find('.//{http://sndlib.zib.de/network}target').text

        gml_graph.add_edge(source, target, id=link_id)

        for add_module in link.findall('.//{http://sndlib.zib.de/network}addModule'):
            capacity = float(add_module.find('.//{http://sndlib.zib.de/network}capacity').text)
            cost = float(add_module.find('.//{http://sndlib.zib.de/network}cost').text)
            gml_graph[source][target]['capacity'] = capacity
            gml_graph[source][target]['cost'] = cost
    # Export the GML graph to the specified file
    nx.write_gml(gml_graph, gml_file_path)

    print("GML data has been written to:", gml_file_path)

# Usage example:
xml_input_file = "./sndlib-networks-xml/brain.xml"
gml_output_file = "./data/PHY/brain.gml"
xml_to_gml(xml_input_file, gml_output_file)
