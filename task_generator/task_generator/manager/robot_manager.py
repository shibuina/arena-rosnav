import dataclasses
import typing

import numpy as np
import os
import scipy.spatial.transform
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from task_generator.constants import Constants
from task_generator.constants.runtime import Config
from task_generator.manager.entity_manager.entity_manager import EntityManager
from task_generator.manager.entity_manager.utils import YAMLUtil
from task_generator.shared import ModelType, Namespace, PositionOrientation, Robot

import task_generator.utils.arena as Utils
from task_generator.utils.geometry import quaternion_from_euler

import nav_msgs.msg as nav_msgs
import geometry_msgs.msg as geometry_msgs
import std_srvs.srv as std_srvs

import launch
from launch import LaunchDescription
from launch_ros.actions import Node as LaunchNode
from launch.launch_service import LaunchService

class RobotManager(Node):
    """
    The robot manager manages the goal and start
    position of a robot for all task modes.
    """

    ## Define class attributes with type annotations
    _namespace: Namespace
    _entity_manager: EntityManager
    _start_pos: PositionOrientation
    _goal_pos: PositionOrientation
    _position: PositionOrientation
    _robot_radius: float
    _goal_tolerance_distance: float
    _goal_tolerance_angle: float
    _robot: Robot
    _move_base_goal_pub: rclpy.publisher.Publisher
    _pub_goal_timer: rclpy.timer.Timer
    _clear_costmaps_srv: rclpy.client.Client

    @property
    def start_pos(self) -> PositionOrientation:
        return self._start_pos

    @property
    def goal_pos(self) -> PositionOrientation:
        return self._goal_pos

    def __init__(
        self, namespace: Namespace, entity_manager: EntityManager, robot: Robot
    ):
        ## Initialize the Node
        super().__init__('robot_manager')
        self._namespace = namespace
        self._entity_manager = entity_manager
        self._start_pos = PositionOrientation(0, 0, 0)
        self._goal_pos = PositionOrientation(0, 0, 0)

        ## Declare parameters with default values
        self.declare_parameter('goal_radius', Config.Robot.GOAL_TOLERANCE_RADIUS)
        self.declare_parameter('goal_tolerance_angle', Config.Robot.GOAL_TOLERANCE_ANGLE)
        self.declare_parameter(f'{robot.name}/safety_distance', Config.Robot.SPAWN_ROBOT_SAFE_DIST)

        ## Initialize configuration using parameters
        self._goal_tolerance_distance = self.get_parameter('goal_radius').get_parameter_value().double_value
        self._goal_tolerance_angle = self.get_parameter('goal_tolerance_angle').get_parameter_value().double_value
        self._safety_distance = self.get_parameter(f'{robot.name}/safety_distance').get_parameter_value().double_value

        self._robot = robot
        self._position = self._start_pos

        ## Set up parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params: typing.List[Parameter]):
        ## Callback to handle parameter updates
        for param in params:
            if param.name == 'goal_radius':
                self._goal_tolerance_distance = param.value
            elif param.name == 'goal_tolerance_angle':
                self._goal_tolerance_angle = param.value
            elif param.name == f'{self._robot.name}/safety_distance':
                self._safety_distance = param.value
        return rclpy.parameter.SetParametersResult(successful=True)

    def set_up_robot(self):
        ## Set up the robot model and entity manager
        self._robot = dataclasses.replace(
            self._robot,
            model=self._robot.model.override(
                model_type=ModelType.YAML,
                override=lambda model: model.replace(
                    description=YAMLUtil.serialize(
                        YAMLUtil.update_plugins(
                            namespace=self.namespace,
                            description=YAMLUtil.parse_yaml(model.description),
                        )
                    )
                ),
            ),
        )

        self._entity_manager.spawn_robot(self._robot)

        _gen_goal_topic = self.namespace("move_base_simple", "goal")

        ## Create publisher for the goal topic
        self._move_base_goal_pub = self.create_publisher(
            geometry_msgs.PoseStamped, _gen_goal_topic, 10
        )

        ## Create timer to publish goals periodically
        self._pub_goal_timer = self.create_timer(
            0.25, self._publish_goal_periodically
        )

        ## Subscribe to odometry data
        self.create_subscription(
            nav_msgs.Odometry, self.namespace("odom"), self._robot_pos_callback, 10
        )

        self._launch_robot()

        ## Set robot radius based on the arena type
        if Utils.get_arena_type() == Constants.ArenaType.TRAINING:
            self._robot_radius = float(self.get_parameter("robot_radius").get_parameter_value().double_value)
        else:
            self.declare_parameter(self.namespace("robot_radius"), Config.Robot.GOAL_TOLERANCE_RADIUS)
            self._robot_radius = self.get_parameter(self.namespace("robot_radius")).get_parameter_value().double_value

        ## Set up service client for clearing costmaps
        self._clear_costmaps_srv = self.create_client(
            std_srvs.Empty, self.namespace("move_base", "clear_costmaps")
        )

    @property
    def safe_distance(self) -> float:
        return self._robot_radius + self._safety_distance

    @property
    def model_name(self) -> str:
        return self._robot.model.name

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def namespace(self) -> Namespace:
        ## Return the appropriate namespace based on the arena type
        if Utils.get_arena_type() == Constants.ArenaType.TRAINING:
            return Namespace(
                f"{self._namespace}{self._namespace}_{self.model_name}"
            )

        return self._namespace(self._robot.name)

    @property
    def is_done(self) -> bool:
        ## Check if the robot has reached the goal
        return self._is_goal_reached

    def move_robot_to_pos(self, position: PositionOrientation):
        ## Move the robot to the specified position
        self._entity_manager.move_robot(name=self.name, position=position)

    def reset(
        self,
        start_pos: typing.Optional[PositionOrientation],
        goal_pos: typing.Optional[PositionOrientation],
    ):
        ## Reset the robot's start and goal positions
        if start_pos is not None:
            self._start_pos = start_pos
            self.move_robot_to_pos(start_pos)

            if self._robot.record_data_dir is not None:
                self.set_parameter(
                    rclpy.parameter.Parameter(
                        self.namespace("start"),
                        rclpy.Parameter.Type.DOUBLE_ARRAY,
                        [float(v) for v in self._start_pos]
                    )
                )

        if goal_pos is not None:
            self._goal_pos = goal_pos
            self._publish_goal(self._goal_pos)

            if self._robot.record_data_dir is not None:
                self.set_parameter(
                    rclpy.parameter.Parameter(
                        self.namespace("goal"),
                        rclpy.Parameter.Type.DOUBLE_ARRAY,
                        [float(v) for v in self._goal_pos]
                    )
                )

        ## Clear costmaps if service is available
        if self._clear_costmaps_srv.wait_for_service(timeout_sec=1.0):
            self._clear_costmaps_srv.call_async(std_srvs.Empty.Request())

        return self._position, self._goal_pos

    @property
    def _is_goal_reached(self) -> bool:
        ## Calculate the distance and angle to the goal
        start = self._position
        goal = self._goal_pos

        distance_to_goal: float = np.linalg.norm(
            np.array(goal[:2]) - np.array(start[:2])
        )

        angle_to_goal: float = np.pi - np.abs(np.abs(goal[2] - start[2]) - np.pi)

        return (
            distance_to_goal < self._goal_tolerance_distance
            and angle_to_goal < self._goal_tolerance_angle
        )

    def _publish_goal_periodically(self):
        ## Publish the goal periodically if available
        if self._goal_pos is not None:
            self._publish_goal(self._goal_pos)

    def _publish_goal(self, goal: PositionOrientation):
        ## Publish the goal position to the topic
        goal_msg = geometry_msgs.PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = goal.x
        goal_msg.pose.position.y = goal.y
        goal_msg.pose.position.z = 0

        goal_msg.pose.orientation = geometry_msgs.Quaternion(
            *quaternion_from_euler(0.0, 0.0, goal.orientation, axes="sxyz")
        )

        self._move_base_goal_pub.publish(goal_msg)

    def _launch_robot(self):
        ## Launch the robot using a launch file
        self.get_logger().warn(f"START WITH MODEL {self.namespace}")

        if Utils.get_arena_type() != Constants.ArenaType.TRAINING:
            launch_description = LaunchDescription()

            launch_arguments = {
                'SIMULATOR': Utils.get_simulator().value,
                'model': self.model_name,
                'name': self.name,
                'namespace': self.namespace,
                'frame': f"{self.name}/" if self.name else '',
                'inter_planner': self._robot.inter_planner,
                'local_planner': self._robot.local_planner,
                'complexity': self.get_parameter('complexity').get_parameter_value().integer_value,
                'train_mode': self.get_parameter('train_mode').get_parameter_value().bool_value,
                'agent_name': self._robot.agent,
            }

            if self._robot.record_data_dir:
                launch_arguments.update({
                    'record_data': 'true',
                    'record_data_dir': self._robot.record_data_dir,
                })

            launch_description.add_action(LaunchNode(
                package='arena_bringup',
                executable='robot',
                output='screen',
                parameters=[launch_arguments]
            ))

            self.launch_service = LaunchService()
            self.launch_service.include_launch_description(launch_description)
            self.launch_service.run()

        base_frame: str = self.get_parameter(self.namespace("robot_base_frame")).get_parameter_value().string_value
        sensor_frame: str = self.get_parameter(self.namespace("robot_sensor_frame")).get_parameter_value().string_value

        self.set_parameters([
            Parameter(self.namespace("move_base", "global_costmap", "robot_base_frame"), Parameter.Type.STRING, os.path.join(self.name, base_frame)),
            Parameter(self.namespace("move_base", "local_costmap", "robot_base_frame"), Parameter.Type.STRING, os.path.join(self.name, base_frame)),
            Parameter(self.namespace("move_base", "local_costmap", "scan", "sensor_frame"), Parameter.Type.STRING, os.path.join(self.name, sensor_frame)),
            Parameter(self.namespace("move_base", "global_costmap", "scan", "sensor_frame"), Parameter.Type.STRING, os.path.join(self.name, sensor_frame))
        ])

    def _robot_pos_callback(self, data: nav_msgs.Odometry):
        ## Callback to update robot's current position based on odometry data
        current_position = data.pose.pose
        quat = current_position.orientation

        rot = scipy.spatial.transform.Rotation.from_quat(
            [quat.x, quat.y, quat.z, quat.w]
        )

        self._position = PositionOrientation(
            current_position.position.x,
            current_position.position.y,
            rot.as_euler("xyz")[2],
        )

## Main function to start the node
def main(args=None):
    rclpy.init(args=args)
    node = RobotManager()  ## Initialize RobotManager node
    rclpy.spin(node)  
    rclpy.shutdown()  
