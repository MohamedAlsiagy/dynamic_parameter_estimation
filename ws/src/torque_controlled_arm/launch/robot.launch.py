from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch arguments
    urdf_path_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value=PathJoinSubstitution([
            get_package_share_directory('torque_controlled_arm'),
            'robots',
            'robot_0',
            'robotGA.urdf'
        ]),
        description='Path to the URDF file'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value="robot_0",
        description='name of robot in gazebo'
    )

    x_arg = DeclareLaunchArgument(
        'x',
        default_value='0',
        description='x pose'
    )
    
    y_arg = DeclareLaunchArgument(
        'y',
        default_value='0',
        description='y pose'
    )
    
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Config paths
    pkg_share = get_package_share_directory('torque_controlled_arm')
    urdf_path = LaunchConfiguration('urdf_path')
    robot_name = LaunchConfiguration('robot_name')

    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')

    robot_desc = Command(['xacro ', urdf_path, ' namespace:=', robot_name ])

    run_controller_editor = ExecuteProcess(
        cmd=[
            'python3',
            '/root/ws/src/torque_controlled_arm/utils/controller_adapter.py',
            robot_name,
        ],
        shell=False,
        output='screen'
    )

    run_urdf_editor = ExecuteProcess(
        cmd=[
            'python3',
            '/root/ws/src/torque_controlled_arm/utils/robot_controller_editor.py',
            urdf_path,
            robot_name,
        ],
        shell=False,
        output='screen'
    )

    # Robot State Publisher with namespace
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=robot_name,  # Add namespace
        output='both',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'frame_prefix': robot_name
        }]
    )

    # Spawn Robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', [robot_name, '/robot_description'],  # Namespaced topic
            '-name', robot_name,
            '-z', '0',
            '-x', x,
            '-y', y,
        ],
        output='screen'
    )

    # Load controllers with namespace
    load_joint_state_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster',
             '-c', ['/', robot_name, '/controller_manager']],  # Namespaced controller manager
        output='screen',
        shell=True
    )

    load_torque_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'torque_arm_controller',
             '-c', ['/', robot_name, '/controller_manager']],  # Namespaced controller manager
        output='screen',
        shell=True
    )

    load_joint_state_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot,
            on_exit=[load_joint_state_controller]
        )
    )

    load_torque_controller_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_joint_state_controller,
            on_exit=[load_torque_controller]
        )
    )

    return LaunchDescription([
        use_sim_time,
        urdf_path_arg,
        robot_name_arg,
        x_arg,
        y_arg,
        run_controller_editor,
        run_urdf_editor,
        robot_state_publisher,
        spawn_robot,
        load_joint_state_event,
        load_torque_controller_event,
    ])