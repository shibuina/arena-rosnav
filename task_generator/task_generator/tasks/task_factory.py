import os
import typing

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from rosros import rospify as rospy
from rospkg import RosPack

from task_generator.constants import Constants
from task_generator.manager.obstacle_manager import ObstacleManager
from task_generator.manager.robot_manager import RobotManager
from task_generator.manager.world_manager import WorldManager
from task_generator.shared import PositionOrientation
from task_generator.tasks import Task
from task_generator.tasks.modules import TM_Module
from task_generator.tasks.obstacles import TM_Obstacles
from task_generator.tasks.robots import TM_Robots

import std_msgs.msg as std_msgs
import rosgraph_msgs.msg as rosgraph_msgs

from task_generator.utils import ModelLoader


class TaskFactory(Node):
    registry_obstacles: typing.Dict[Constants.TaskMode.TM_Obstacles, typing.Callable[[], typing.Type[TM_Obstacles]]] = {}
    registry_robots: typing.Dict[Constants.TaskMode.TM_Robots, typing.Callable[[], typing.Type[TM_Robots]]] = {}
    registry_module: typing.Dict[Constants.TaskMode.TM_Module, typing.Callable[[], typing.Type[TM_Module]]] = {}

    def __init__(self):
        super().__init__('task_factory')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('tm_robots', ''),
                ('tm_obstacles', ''),
                ('train_mode', False)
            ]
        )

        self._train_mode = self.get_parameter('train_mode').get_parameter_value().bool_value
        self._force_reset = False
        self.__reset_mutex = False

        # Publishers
        self.__reset_start = self.create_publisher(std_msgs.Empty, 'reset_start', 1)
        self.__reset_end = self.create_publisher(std_msgs.Empty, 'reset_end', 1)

        # Subscribers
        self.create_subscription(rosgraph_msgs.Clock, '/clock', self._clock_callback, 10)
        self.clock = rosgraph_msgs.Clock()
        self.last_reset_time = 0

        # Initialize the rest as None, will be set later
        self.__tm_robots = None
        self.__tm_obstacles = None

        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params: typing.List[Parameter]):
        for param in params:
            if param.name == 'tm_robots':
                new_tm_robots = Constants.TaskMode.TM_Robots(param.value)
                if new_tm_robots != self.__param_tm_robots:
                    self.set_tm_robots(new_tm_robots)
            elif param.name == 'tm_obstacles':
                new_tm_obstacles = Constants.TaskMode.TM_Obstacles(param.value)
                if new_tm_obstacles != self.__param_tm_obstacles:
                    self.set_tm_obstacles(new_tm_obstacles)
        return rclpy.parameter.SetParametersResult(successful=True)

    @classmethod
    def register_obstacles(cls, name: Constants.TaskMode.TM_Obstacles):
        def inner_wrapper(loader: typing.Callable[[], typing.Type[TM_Obstacles]]):
            assert (
                name not in cls.registry_obstacles
            ), f"TaskMode '{name}' for obstacles already exists!"
            cls.registry_obstacles[name] = loader
            return loader
        return inner_wrapper

    @classmethod
    def register_robots(cls, name: Constants.TaskMode.TM_Robots):
        def inner_wrapper(loader: typing.Callable[[], typing.Type[TM_Robots]]):
            assert (
                name not in cls.registry_obstacles
            ), f"TaskMode '{name}' for robots already exists!"
            cls.registry_robots[name] = loader
            return loader
        return inner_wrapper

    @classmethod
    def register_module(cls, name: Constants.TaskMode.TM_Module):
        def inner_wrapper(loader: typing.Callable[[], typing.Type[TM_Module]]):
            assert (
                name not in cls.registry_obstacles
            ), f"TaskMode '{name}' for module already exists!"
            cls.registry_module[name] = loader
            return loader
        return inner_wrapper

    @classmethod
    def combine(cls, modules: typing.List[Constants.TaskMode.TM_Module] = []) -> typing.Type[Task]:
        for module in modules:
            assert (
                module in cls.registry_module
            ), f"Module '{module}' is not registered!"

        class CombinedTask(Task):
            """
            Represents a combined task that involves multiple robots and obstacles.
            """

            PARAM_TM_ROBOTS = "tm_robots"
            PARAM_TM_OBSTACLES = "tm_obstacles"

            __param_tm_robots: Constants.TaskMode.TM_Robots
            __param_tm_obstacles: Constants.TaskMode.TM_Obstacles

            __tm_robots: TM_Robots
            __tm_obstacles: TM_Obstacles

            def __init__(
                self,
                obstacle_manager: ObstacleManager,
                robot_managers: typing.List[RobotManager],
                world_manager: WorldManager,
                namespace: str = "",
                *args,
                **kwargs,
            ):
                super().__init__(namespace)
                self.namespace = namespace

                self.obstacle_manager = obstacle_manager
                self.robot_managers = robot_managers
                self.world_manager = world_manager

                self.model_loader = ModelLoader(
                    os.path.join(
                        RosPack().get_path("arena_simulation_setup"),
                        "entities",
                        "obstacles",
                        "static",
                    )
                )
                self.dynamic_model_loader = ModelLoader(
                    os.path.join(
                        RosPack().get_path("arena_simulation_setup"),
                        "entities",
                        "obstacles",
                        "dynamic",
                    )
                )

                self.__param_tm_obstacles = None
                self.__param_tm_robots = None
                self.__modules = [
                    cls.registry_module[module]()(task=self) for module in modules
                ]

                if self._train_mode:
                    self.set_tm_robots(Constants.TaskMode.TM_Robots(self.get_parameter('tm_robots').get_parameter_value().string_value))
                    self.set_tm_obstacles(Constants.TaskMode.TM_Obstacles(self.get_parameter('tm_obstacles').get_parameter_value().string_value))

            def set_tm_robots(self, tm_robots: Constants.TaskMode.TM_Robots):
                """
                Sets the task mode for robots.

                Args:
                    tm_robots (Constants.TaskMode.TM_Robots): The task mode for robots.
                """
                assert (
                    tm_robots in cls.registry_robots
                ), f"TaskMode '{tm_robots}' for robots is not registered!"
                self.__tm_robots = cls.registry_robots[tm_robots]()(props=self)
                self.__param_tm_robots = tm_robots

            def set_tm_obstacles(self, tm_obstacles: Constants.TaskMode.TM_Obstacles):
                """
                Sets the task mode for obstacles.

                Args:
                    tm_obstacles (Constants.TaskMode.TM_Obstacles): The task mode for obstacles.
                """
                assert (
                    tm_obstacles in cls.registry_obstacles
                ), f"TaskMode '{tm_obstacles}' for obstacles is not registered!"
                self.__tm_obstacles = cls.registry_obstacles[tm_obstacles]()(props=self)
                self.__param_tm_obstacles = tm_obstacles

            def _reset_task(self, **kwargs):
                """
                Reset the task by updating task modes, resetting modules, and spawning obstacles.

                Args:
                    **kwargs: Additional keyword arguments for resetting the task.
                """
                try:
                    self.__reset_start.publish(std_msgs.Empty())

                    if not self._train_mode:
                        if (
                            new_tm_robots := Constants.TaskMode.TM_Robots(
                                self.get_parameter(self.PARAM_TM_ROBOTS).get_parameter_value().string_value
                            )
                        ) != self.__param_tm_robots:
                            self.set_tm_robots(new_tm_robots)

                        if (
                            new_tm_obstacles := Constants.TaskMode.TM_Obstacles(
                                self.get_parameter(self.PARAM_TM_OBSTACLES).get_parameter_value().string_value
                            )
                        ) != self.__param_tm_obstacles:
                            self.set_tm_obstacles(new_tm_obstacles)

                    for module in self.__modules:
                        module.before_reset()

                    self.__tm_robots.reset(**kwargs)
                    obstacles, dynamic_obstacles = self.__tm_obstacles.reset(**kwargs)

                    def respawn():
                        self.obstacle_manager.spawn_obstacles(obstacles)
                        self.obstacle_manager.spawn_dynamic_obstacles(dynamic_obstacles)

                    self.obstacle_manager.respawn(respawn)

                    for module in self.__modules:
                        module.after_reset()

                    self.last_reset_time = self.clock.clock.sec

                except Exception as e:
                    self.get_logger().error(repr(e))
                    rclpy.shutdown("Reset error!")
                    raise Exception("reset error!") from e

                finally:
                    self.__reset_end.publish(std_msgs.Empty())

            def _mutex_reset_task(self, **kwargs):
                """
                Executes a reset task while ensuring mutual exclusion.
                """
                while self.__reset_mutex:
                    rclpy.sleep(0.001)
                self.__reset_mutex = True

                try:
                    self._reset_task()

                except Exception as e:
                    raise e

                finally:
                    self.__reset_mutex = False

            def reset(self, **kwargs):
                """
                Resets the task.

                Args:
                    **kwargs: Arbitrary keyword arguments.
                """
                self._force_reset = False
                if self._train_mode:
                    self._reset_task(**kwargs)
                else:
                    self._mutex_reset_task(**kwargs)

            @property
            def is_done(self) -> bool:
                """
                Checks if the task is done.

                Returns:
                    bool: True if the task is done, False otherwise.
                """
                return self._force_reset or self.__tm_robots.done

            def set_robot_position(self, position: PositionOrientation):
                """
                Sets the position of the robot.

                Args:
                    position (PositionOrientation): The position and orientation of the robot.
                """
                self.__tm_robots.set_position(position)

            def set_robot_goal(self, position: PositionOrientation):
                """
                Sets the goal position for the robot.

                Args:
                    position (PositionOrientation): The goal position for the robot.
                """
                self.__tm_robots.set_goal(position)

            def force_reset(self):
                self._force_reset = True

        return CombinedTask


def main(args=None):
    rclpy.init(args=args)
    task_factory = TaskFactory()  # Initialize TaskFactory node
    rclpy.spin(task_factory)  
    rclpy.shutdown()  