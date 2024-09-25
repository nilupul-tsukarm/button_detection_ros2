from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='button_detection',
            executable='button_detection_node',
            output='screen'
        )
    ])
