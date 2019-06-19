import xml.etree.ElementTree as ET
import numpy as np
import Astar
tree = ET.parse('map1.net.xml')
root = tree.getroot()
graph_dict = {}

key_list_edg_internal = root[1].attrib.keys()
for child in root:
    if child.tag == 'edge':
        if child.attrib.keys() == key_list_edg_internal:
            continue
        else:
            if int(child.attrib['id']) < 0:
                (child.attrib['id']) = -1 * int(child.attrib['id'])
                child.attrib['id'] = str(child.attrib['id'])
            graph_dict[child.attrib['id']] = [child.attrib['from'], child.attrib['to']]
print(graph_dict)

arr = np.zeros((len(graph_dict),len(graph_dict)))
for i in len(graph_dict):


