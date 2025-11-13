import json
import os
from xml.etree.ElementTree import tostring
from xml.dom import minidom

# Function to prettify the XML output
def prettify(xml_str):
    rough_string = tostring(xml_str, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return None
    if os.path.getsize(file_path) == 0:
        print(f"Warning: The file {file_path} is empty.")
        return None
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {file_path}: {e}")
            return None