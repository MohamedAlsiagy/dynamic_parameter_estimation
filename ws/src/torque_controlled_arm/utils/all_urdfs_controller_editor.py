from pathlib import Path
from robots_editor import process_all_robot_folders

if __name__ == "__main__":
    robot_dir = Path(__file__).parent.parent / "robots"

    tag_path = [
        {"tag": "gazebo"},
        {"tag": "plugin", "attrs": {"filename": "libign_ros2_control-system.so"}},
        {"tag": "parameters"}
    ]

    def change_controller_path(old: str) -> str:
        return "/root/ws/src/torque_controlled_arm/config/controllers.yaml"

    process_all_robot_folders(robot_dir, tag_path, change_controller_path)
    print("✅ Done updating matching tags.")
