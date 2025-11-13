import xml.etree.ElementTree as ET
from typing import Callable, List, Dict
from pathlib import Path
from tqdm import tqdm

TagSpec = Dict[str, object]  # {'tag': 'robot', 'attrs': {'name': 'robot_0'}}

def find_element_by_tag_path(root: ET.Element, path: List[TagSpec]) -> ET.Element | None:
    """Traverse the XML tree based on tag path with optional attribute filters."""
    elem = root
    for spec in path:
        tag = spec["tag"]
        attrs = spec.get("attrs", {})
        found = None
        for child in elem.findall(tag):
            if all(child.attrib.get(k) == v for k, v in attrs.items()):
                found = child
                break
        if found is None:
            return None
        elem = found
    return elem

def modify_tag_text(
    input_urdf_path: Path,
    tag_path: List[TagSpec],
    modify_fn: Callable[[str], str],
    output_urdf_path: Path = None
):
    tree = ET.parse(input_urdf_path)
    root = tree.getroot()

    target_elem = find_element_by_tag_path(root, tag_path)
    if target_elem is None:
        print(f"[WARNING] Tag path not found in {input_urdf_path.name}")
        return

    old_text = target_elem.text or ""
    new_text = modify_fn(old_text)
    target_elem.text = new_text

    tree.write(output_urdf_path or input_urdf_path, encoding='utf-8', xml_declaration=True)
    # print(f"[INFO] Updated text in: {input_urdf_path.name}")

def process_all_robot_folders(base_dir: Path, tag_path: List[TagSpec], modify_fn: Callable[[str], str]):
    base_path = Path(base_dir)
    robot_folders = [f for f in base_path.iterdir() if f.is_dir()]

    for folder in tqdm(robot_folders, desc="Processing robots", unit="robot"):
        urdf_file = folder / "robotGA.urdf"
        if not urdf_file.exists():
            print(f"[WARNING] Skipping {folder.name} (no URDF found)")
            continue

        modify_tag_text(
            input_urdf_path=urdf_file,
            tag_path=tag_path,
            modify_fn=modify_fn
        )



