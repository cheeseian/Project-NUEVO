"""
test_orientation_fusion.launch.py
==================================
Starts the robot node using test_orientation_fusion.py as the run() entry point
instead of main.py.

The robot will drive two circles, then stop and save a trajectory/heading plot
to ~/fusion_test_result.png.

Launch arguments:
    launch_robot_node  (default: true)  — set false if the robot node is
                                          already running and you only want
                                          to attach the test runner
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_robot_node",
                default_value="true",
                description=(
                    "Set to false if the robot node is already running and you "
                    "only want to attach the test runner."
                ),
            ),
            Node(
                package="robot",
                executable="test_orientation_fusion",
                name="robot",
                output="screen",
                condition=IfCondition(LaunchConfiguration("launch_robot_node")),
            ),
        ]
    )
