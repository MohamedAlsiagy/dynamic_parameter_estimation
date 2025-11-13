import os
import random
import colorsys
from pathlib import Path
from xml.etree import ElementTree as ET
import math

def interpolate_color(c1, c2, t):
    return [(1 - t) * a + t * b for a, b in zip(c1, c2)]

def rgba_string(color):
    return " ".join(f"{v:.3f}" for v in color) + " 1"

def generate_vibrant_color():
    h = random.random()              # Random hue
    s = random.uniform(0.8, 1.0)     # High saturation
    l = random.uniform(0.4, 0.6)     # Medium lightness for vibrancy
    return colorsys.hls_to_rgb(h, l, s)

def clean_and_format_urdf(folder_path):
    urdf_path = folder_path / "robot.urdf"
    if not urdf_path.exists():
        print(f"Skipped {folder_path}: no robot.urdf found.")
        return

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Remove all <transmission> blocks
    for trans in root.findall('transmission'):
        root.remove(trans)

    # Replace "dae/" with "stl/", and prepend full path to stl/ files
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if "dae/" in filename:
            mesh.set("filename", filename.replace("dae", "stl"))
        filename = mesh.get("filename", "")
        if filename.startswith("stl/"):
            mesh.set("filename", str(folder_path / filename))

    # Add world link/joint
    world_link = ET.fromstring('<link name="world"/>')
    world_joint = ET.fromstring('''
    <joint name="world_to_base" type="fixed">
      <parent link="world"/>
      <child link="link_0"/>
    </joint>''')

    # Get all joints and links
    joints = [j.get("name") for j in root.findall("joint")]
    num_links = len(root.findall("link"))

    # Generate two vibrant RGB colors
    c1 = generate_vibrant_color()
    c2 = generate_vibrant_color()

    # Gazebo plugin
    gazebo_plugin = ET.fromstring('''
    <gazebo>
      <plugin filename="libign_ros2_control-system.so" name="ign_ros2_control::IgnitionROS2ControlPlugin">
        <parameters>/root/ws/src/torque_controlled_arm/config/controllers.yaml</parameters>
        <controller_manager_node_name>/controller_manager</controller_manager_node_name>
      </plugin>
    </gazebo>''')

    # Prepare blocks
    gazebo_tags = []
    transmission_tags = []

    ros2_control = ET.Element("ros2_control", {"name": "robot_0_control", "type": "system"})
    hardware = ET.SubElement(ros2_control, "hardware")
    plugin = ET.SubElement(hardware, "plugin")
    plugin.text = "ign_ros2_control/IgnitionSystem"

    # Final reordered elements
    reordered = [world_link, world_joint, gazebo_plugin]

    for i in range(num_links):
        color = interpolate_color(c1, c2, i / max(1, num_links - 1))
        color_rgba = rgba_string(color)
        mat_name = f"color_{i}"

        # Create material
        material_tag = ET.fromstring(f'''
        <material name="{mat_name}">
          <color rgba="{color_rgba}"/>
        </material>''')
        reordered.append(material_tag)

        # Link
        link_tag = root.find(f"link[@name='link_{i}']")
        if link_tag is not None:
            # Add material to visual
            visual = link_tag.find("visual")
            if visual is not None and visual.find("material") is None:
                visual.append(ET.fromstring(f'<material name="{mat_name}"/>'))
            reordered.append(link_tag)

        if i != num_links - 1:
          # Joint
          joint_tag = root.find(f"joint[@name='joint_{i}']")
          if joint_tag is not None:
              # Remove existing tags
              for tag_name in ['limit', 'dynamics']:
                  old_tag = joint_tag.find(tag_name)
                  if old_tag is not None:
                      joint_tag.remove(old_tag)

              # Add new limit
              limit_tag = ET.Element("limit", {
                  "lower": str(2 * -math.pi),
                  "upper": str(2 * math.pi),
                  "effort": "8000.0",
                  "velocity": "30.0"
              })
              joint_tag.append(limit_tag)

              # Add new dynamics
              dynamics_tag = ET.Element("dynamics", {
                  "damping": "0.1",
                  "friction": "0.4"
              })
              joint_tag.append(dynamics_tag)
              reordered.append(joint_tag)

          # Transmission
          joint_name = f"joint_{i}"
          trans_block = ET.fromstring(f'''
          <transmission name="transmission_{joint_name}">
            <type>transmission_interface/SimpleTransmission</type>
            <joint name="{joint_name}">
              <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
            </joint>
            <actuator name="motor_{joint_name}">
              <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
              <mechanicalReduction>1</mechanicalReduction>
            </actuator>
          </transmission>''')
          transmission_tags.append(trans_block)

          gazebo_joint = ET.fromstring(f'''
          <gazebo reference="{joint_name}">
            <implicitSpringDamper>true</implicitSpringDamper>
          </gazebo>''')
          gazebo_tags.append(gazebo_joint)

          # ROS 2 Control
          ros2_joint = ET.SubElement(ros2_control, "joint", {"name": joint_name})
          ET.SubElement(ros2_joint, "command_interface", {"name": "effort"})
          for iface in ["position", "velocity", "effort"]:
              ET.SubElement(ros2_joint, "state_interface", {"name": iface})

          reordered.append(trans_block)
          reordered.append(gazebo_joint)

    reordered.append(ros2_control)

    # Clear and rebuild
    for elem in list(root):
        root.remove(elem)
    for elem in reordered:
        root.append(elem)

    # Save output
    output_path = folder_path / "robotGA.urdf"
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def process_robot(folder):
    if folder.is_dir():
        clean_and_format_urdf(folder)

def process_robot_pack(base_dir):
    base_path = Path(base_dir)
    for folder in base_path.iterdir():
        process_robot(folder)
        
process_robot_pack("/root/ws/src/torque_controlled_arm/robots")
print("urdfs have been adapted")