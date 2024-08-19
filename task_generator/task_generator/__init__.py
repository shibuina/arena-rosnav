import dataclasses
import os
import traceback
from typing import Dict, List

import rclpy
import rclpy.node
import rclpy.executors
from rclpy.parameter import Parameter

import yaml
from ament_index_python.packages import get_package_share_directory
from task_generator.constants import Constants
from task_generator.constants.runtime import Config

from task_generator.manager.entity_manager.entity_manager import EntityManager
from task_generator.manager.world_manager import WorldManager
from task_generator.manager.obstacle_manager import ObstacleManager
from task_generator.shared import (
    ModelWrapper,
    Namespace,
    Robot,
    gen_init_pos,
)

from task_generator.simulators import BaseSimulator, SimulatorFactory
from task_generator.tasks.task_factory import TaskFactory
from task_generator.utils import ModelLoader
import task_generator.utils.arena as Utils

import map_distance_server.srv as map_distance_server_srvs
from std_msgs.msg import Int16, Empty
import std_srvs.srv as std_srvs
from std_srvs.srv import Empty as EmptySrv

def create_default_robot_list(
    robot_model: ModelWrapper,
    name: str,
    inter_planner: str,
    local_planner: str,
    agent: str
) -> List[Robot]:
    return [
        Robot(
            model=robot_model,
            inter_planner=inter_planner,
            local_planner=local_planner,
            agent=agent,
            position=next(gen_init_pos),
            name=name,
            record_data_dir=Config.Robot.RECORD_DATA_DIR,
            extra=dict(),
        )
    ]

def read_robot_setup_file(setup_file: str) -> List[Dict]:
    try:
        with open(
            os.path.join(
                get_package_share_directory("arena_bringup"),
                "configs",
                "robot_setup",
                setup_file,
            ),
            "r",
        ) as f:
            robots: List[Dict] = yaml.safe_load(f)["robots"]

        return robots

    except Exception as e:
        traceback.print_exc()
        raise Exception("Failed to read robot setup file") from e

class TaskGenerator(rclpy.node.Node):
    """
    Task Generator Node
    Will initialize and reset all tasks. The task to use is read from the `/task_mode` param.
    """

    def __init__(self, namespace: str = "/"):
        super().__init__('task_generator')
        self._namespace = Namespace(namespace)

        # Declare all parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('entity_manager', ''),
                ('auto_reset', True),
                ('train_mode', False),
                ('robot_setup_file', ''),
                ('model', ''),
                ('inter_planner', ''),
                ('local_planner', ''),
                ('agent_name', ''),
                ('robot_names', []),
                ('task_generator_setup_finished', False),
            ]
        )
        
        # Initialize parameters
        self._auto_reset = self.get_parameter('auto_reset').get_parameter_value().bool_value
        self._train_mode = self.get_parameter('train_mode').get_parameter_value().bool_value

        # Publishers
        if not self._train_mode:
            self._pub_scenario_reset = self.create_publisher(Int16, 'scenario_reset', 1)
            self._pub_scenario_finished = self.create_publisher(Empty, 'scenario_finished', 10)

            # Services
            self.create_service(EmptySrv, 'reset_task', self._reset_task_srv_callback)

        # Parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params: List[Parameter]):
        """
        Handle dynamic updates to parameters
        """
        for param in params:
            if param.name == 'auto_reset':
                self._auto_reset = param.value
            elif param.name == 'train_mode':
                self._train_mode = param.value
        return rclpy.parameter.SetParametersResult(successful=True)

    def post_init(self):
        """
        Initialize other components and start the node spinning
        """
        # Initialize the simulator
        self._env_wrapper = SimulatorFactory.instantiate(Utils.get_simulator())(
            namespace=self._namespace
        )

        # Loaders
        self._robot_loader = ModelLoader(
            os.path.join(get_package_share_directory("arena_simulation_setup"), "entities", "robots")
        )

        if not self._train_mode:
            self._start_time = self.get_clock().now().seconds_nanoseconds()[0]
            self._task = self._get_predefined_task()
            self.set_parameters([rclpy.Parameter('robot_names', value=self._task.robot_names)])

            self._number_of_resets = 0

            self.srv_start_model_visualization = self.create_client(EmptySrv, 'start_model_visualization')
            while not self.srv_start_model_visualization.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('start_model_visualization service not available, waiting again...')
            self.srv_start_model_visualization.call_async(EmptySrv.Request())

            self.create_timer(0.5, self._check_task_status)

            self.reset_task(first_map=True)

            try:
                self.set_parameters([rclpy.Parameter('task_generator_setup_finished', value=True)])
                self.srv_setup_finished = self.create_client(EmptySrv, 'task_generator_setup_finished')
                while not self.srv_setup_finished.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('task_generator_setup_finished service not available, waiting again...')
                self.srv_setup_finished.call_async(EmptySrv.Request())
            except:
                pass

    def _get_predefined_task(self, **kwargs):
        """
        Gets the task based on the passed mode
        """
        if self._env_wrapper is None:
            self._env_wrapper = SimulatorFactory.instantiate(Utils.get_simulator())(
                self._namespace
            )

        cli = self.create_client(map_distance_server_srvs.GetDistanceMap, '/distance_map')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        req = map_distance_server_srvs.GetDistanceMap.Request()
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        map_response = future.result()
        world_manager = WorldManager(
            world_map=WorldMap.from_occupancy_grid(occupancy_grid=map_response)
        )

        self._entity_manager = EntityManager(
            namespace=self._namespace, simulator=self._env_wrapper
        )

        obstacle_manager = ObstacleManager(
            namespace=self._namespace,
            world_manager=world_manager,
            simulator=self._env_wrapper,
            entity_manager=self._entity_manager,
        )

        obstacle_manager.spawn_world_obstacles(world_manager.world)

        robot_managers = self._create_robot_managers()

        tm_modules_value = self.get_parameter("tm_modules").get_parameter_value().string_value
        tm_modules = list(
            set(
                [
                    Constants.TaskMode.TM_Module(mod)
                    for mod in tm_modules_value.split(",")
                    if mod != ""
                ]
            )
        )

        tm_modules.append(Constants.TaskMode.TM_Module.CLEAR_FORBIDDEN_ZONES)
        tm_modules.append(Constants.TaskMode.TM_Module.RVIZ_UI)

        if self.get_parameter("map_file").get_parameter_value().string_value == "dynamic_map":
            tm_modules.append(Constants.TaskMode.TM_Module.DYNAMIC_MAP)

        self.get_logger().debug("utils calls task factory")
        task = TaskFactory.combine(
            modules=[Constants.TaskMode.TM_Module(module) for module in tm_modules]
        )(
            obstacle_manager=obstacle_manager,
            robot_managers=robot_managers,
            world_manager=world_manager,
            namespace=self._namespace,
            **kwargs,
        )

        return task

    def _create_robot_managers(self) -> List[RobotManager]:
        """
        Creates robot managers based on configuration
        """
        robot_setup_file: str = self.get_parameter('robot_setup_file').get_parameter_value().string_value

        robot_model: str = self.get_parameter('model').get_parameter_value().string_value

        if robot_setup_file == "":
            robots = create_default_robot_list(
                robot_model=self._robot_loader.bind(robot_model),
                inter_planner=self.get_parameter("inter_planner").get_parameter_value().string_value,
                local_planner=self.get_parameter("local_planner").get_parameter_value().string_value,
                agent=self.get_parameter("agent_name").get_parameter_value().string_value,
                name=f"{self._namespace[1:]}_{robot_model}"
                if self._train_mode
                else robot_model,
            )
        else:
            robots = [
                dataclasses.replace(
                    Robot.parse(
                        robot,
                        model=self._robot_loader.bind(robot["model"]),
                    ),
                    name=f'{robot["model"]}_{i}_{robot.get("amount", 1)-1}'
                )
                for robot in read_robot_setup_file(robot_setup_file)
                for i in range(robot.get("amount", 1))
            ]

        if Utils.get_arena_type() == Constants.ArenaType.TRAINING:
            return [
                RobotManager(
                    namespace=self._namespace,
                    entity_manager=self._entity_manager,
                    robot=robots[0],
                )
            ]

        robot_managers: List[RobotManager] = []

        for robot in robots:
            robot_managers.append(
                RobotManager(
                    namespace=self._namespace,
                    entity_manager=self._entity_manager,
                    robot=robot,
                )
            )

        return robot_managers

    # RUNTIME

    def reset_task(self, **kwargs):
        """
        Reset the current task
        """
        self._start_time = self.get_clock().now().seconds_nanoseconds()[0]

        self._env_wrapper.before_reset_task()

        self.get_logger().info("resetting")

        is_end = self._task.reset(callback=lambda: False, **kwargs)

        self._env_wrapper.after_reset_task()

        self._pub_scenario_reset.publish(Int16(data=self._number_of_resets))
        self._number_of_resets += 1
        self._send_end_message_on_end()

        self._env_wrapper.after_reset_task()

        self.get_logger().info("=============")
        self.get_logger().info("Task Reset!")
        self.get_logger().info("=============")

    def _check_task_status(self, *args, **kwargs):
        """
        Check the status of the task and reset if done
        """
        if self._task.is_done:
            self.reset_task()

    def _reset_task_srv_callback(self, request: std_srvs.Empty.Request, response: std_srvs.Empty.Response):
        """
        Service callback to reset the task
        """
        self.get_logger().debug("Task Generator received task-reset request!")
        self.reset_task()
        return response

    def _send_end_message_on_end(self):
        """
        Send a message when all tasks are completed
        """
        if self._number_of_resets < Config.General.DESIRED_EPISODES:
            return

        self.get_logger().info(f"Shutting down. All {int(Config.General.DESIRED_EPISODES)} tasks completed")
        rclpy.shutdown()

# Main function to start the node
def main(args=None):
    rclpy.init(args=args)
    task_generator = TaskGenerator()  
    task_generator.post_init()  
    rclpy.spin(task_generator)  
    rclpy.shutdown()  
