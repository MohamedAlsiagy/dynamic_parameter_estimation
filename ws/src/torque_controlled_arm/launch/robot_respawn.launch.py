from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch argument
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value="robot",
        description='name of robot in gazebo'
    )

    # Config paths
    robot_name = LaunchConfiguration('robot_name')

    # Spawn Robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', [robot_name, '/robot_description'],  # Namespaced topic
            '-name', robot_name,
            '-z', '0'
        ],
        output='screen'
    )

    return LaunchDescription([
        robot_name_arg,
        spawn_robot,
    ])