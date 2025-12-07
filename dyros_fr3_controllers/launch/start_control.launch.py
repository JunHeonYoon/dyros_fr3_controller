import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import re

def snake_to_pascal(name: str) -> str:
    """Convert snake_case or lowercase name to PascalCase"""
    parts = re.split(r'[_\s]+', name)
    return ''.join(p.capitalize() for p in parts if p)

# Add the path to the `utils` folder
package_share = get_package_share_directory('dyros_fr3_controllers')
utils_path = os.path.join(package_share, '..', '..', 'lib', 'dyros_fr3_controllers', 'utils')
sys.path.append(os.path.abspath(utils_path))

from launch_utils import load_yaml


def generate_robot_nodes(context):
    config_file = LaunchConfiguration('robot_config_file').perform(context)
    controller_name = LaunchConfiguration('controller_name').perform(context)
    configs = load_yaml(config_file)
    nodes = []
    for item_name, config in configs.items():
        namespace = config['namespace']
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('dyros_fr3_controllers'), 'launch', 'franka.launch.py'
                    ])
                ),
                launch_arguments={
                    'arm_id': str(config['arm_id']),
                    'arm_prefix': str(config['arm_prefix']),
                    'namespace': str(namespace),
                    'urdf_file': str(config['urdf_file']),
                    'robot_ip': str(config['robot_ip']),
                    'load_gripper': str(config['load_gripper']),
                    'use_fake_hardware': str(config['use_fake_hardware']),
                    'fake_sensor_commands': str(config['fake_sensor_commands']),
                    'joint_state_rate': str(config['joint_state_rate']),
                }.items(),
            )
        )
        nodes.append(
            Node(
                package='controller_manager',
                executable='spawner',
                namespace=namespace,
                arguments=[controller_name, '--controller-manager-timeout', '30'],
                parameters=[PathJoinSubstitution([
                    FindPackageShare('dyros_fr3_controllers'), 'config', "controllers.yaml",
                ])],
                output='screen',
            )
        )
    if any(str(config.get('use_rviz', 'false')).lower() == 'true' for config in configs.values()):
        nodes.append(
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['--display-config', PathJoinSubstitution([
                    FindPackageShare('franka_description'), 'rviz', 'visualize_franka.rviz'
                ])],
                output='screen',
            )
        )
    return nodes


def generate_gui_nodes(context):
    use_gui = LaunchConfiguration('use_gui').perform(context)
    if use_gui.lower() not in ('true', '1', 'yes'):
        return []

    ctrl_snake = LaunchConfiguration('controller_name').perform(context)

    # Convert snake_case to PascalCase
    ctrl_pascal = snake_to_pascal(ctrl_snake)

    gui_exec = ctrl_pascal + "QT"

    return [
        Node(
            package='dyros_fr3_controllers',
            executable=gui_exec,
            name=gui_exec,
            output='screen',
            emulate_tty=True,
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('dyros_fr3_controllers'), 'config', 'franka.config.yaml'
            ]),
            description='Path to the robot configuration file to load',
        ),
        DeclareLaunchArgument(
            'controller_name',
            default_value='test_effort_controller',
            description='Name of the controller to spawn (required, default: test_effort_controller)',
        ),
        DeclareLaunchArgument(
            'use_gui',
            default_value='true',
            description='Whether to launch Qt GUI for the controller',
        ),
        OpaqueFunction(function=generate_robot_nodes),
        OpaqueFunction(function=generate_gui_nodes),
    ])
