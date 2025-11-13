from pathlib import Path
from robots_editor import modify_tag_text
import sys

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python robot_controller_editor.py <urdf_path> <namespace>")
        sys.exit(1)

    tag_path = [
        {"tag": "gazebo"},
        {"tag": "plugin", "attrs": {"filename": "libign_ros2_control-system.so"}},
        {"tag": "controller_manager_node_name"}
    ]

    urdf_path = sys.argv[1]
    namespace = sys.argv[2]

    def change_controller_manager_name(old: str) -> str:
        return namespace + "/controller_manager"

    modify_tag_text(urdf_path, tag_path, change_controller_manager_name)

    