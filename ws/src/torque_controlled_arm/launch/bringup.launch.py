from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Path configurations
    pkg_share = get_package_share_directory('torque_controlled_arm')

    # Loading Gazebo
    world = os.path.join(pkg_share, "world/empty_world.sdf")
    ign_gz = LaunchConfiguration('ign_gz', default='True')

    ign_gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory("ros_gz_sim"), "launch"), "/gz_sim.launch.py"]
        ),
        launch_arguments={'gz_args': [world, ' -v4 -r']}.items(),
        condition=IfCondition(ign_gz)
    )

    gz_services = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/default/remove@ros_gz_interfaces/srv/DeleteEntity',
            '/world/default/create@ros_gz_interfaces/srv/SpawnEntity',
            '/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose',
            '/world/default/control@ros_gz_interfaces/srv/ControlWorld', # Add for world control
        ],
        output='screen'
    )
    return LaunchDescription([
        use_sim_time,
        ign_gazebo_node,
        gz_services,
    ])







#ign service -s /world/default/control --reqtype ignition.msgs.WorldControl --reptype ignition.msgs.Boolean --timeout 5000 --req 'reset: {all: true}'


"""
ign service -s /world/shapes/playback/control --reqtype ignition.msgs.LogPlaybackControl --reptype ignition.msgs.Boolean --timeout 5000 --req 'seek: {sec: 2} , pause: false'
ign service -s /world/shapes/playback/control --reqtype ignition.msgs.LogPlaybackControl --reptype ignition.msgs.Boolean --timeout 5000 --req 'seek: {sec: 4} , pause: false'
ign service -s /world/shapes/playback/control --reqtype ignition.msgs.LogPlaybackControl --reptype ignition.msgs.Boolean --timeout 5000 --req 'seek: {sec: 0} , pause: false'

ign service -s /world/shapes/playback/control --reqtype ignition.msgs.LogPlaybackControl --reptype ignition.msgs.Boolean --timeout 5000 --req 'rewind: true'


ign service -s /world/default/control --reqtype ign_msgs.WorldControl --reptype ign_msgs.Boolean --timeout 5000 --req 'reset { all: true }'
ign service -s /world/default/control \
  --reqtype ign_msgs.WorldControl \
  --reptype ign_msgs.Boolean \
  --timeout 5000 \
  --req 'reset { time_only: true iterations_only: true model_only: true }'

"""