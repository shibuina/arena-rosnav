import dataclasses
import os
from typing import Any, List, Type
from rosros import rospify as rospy
from task_generator.constants import Constants

from task_generator.manager.obstacle_manager import ObstacleManager
from task_generator.manager.robot_manager import RobotManager
from task_generator.manager.world_manager import WorldManager
from task_generator.shared import Namespace, PositionOrientation

from task_generator.utils import ModelLoader

import rosgraph_msgs.msg as rosgraph_msgs
import std_msgs.msg as std_msgs

import rclpy.node


class Props_Manager:
    obstacle_manager: ObstacleManager
    robot_managers: List[RobotManager]
    world_manager: WorldManager


class Props_Modelloader:
    model_loader: ModelLoader
    dynamic_model_loader: ModelLoader


class Props_Namespace:
    namespace: str
    namespace_prefix: str


class Props_(Props_Manager, Props_Modelloader, Props_Namespace):
    clock: rosgraph_msgs.Clock


class Reconfigurable:
    """
    A class representing a reconfigurable object.

    Attributes:
        NODE_CONFIGURATION (Namespace): The namespace for the task generator server.
    """

    # TOPIC_RECONFIGURE = "RECONFIGURE"

    def reconfigure(self, config):
        ...

    NODE_CONFIGURATION = Namespace(
        os.path.join(rospy.get_namespace(), Constants.TASK_GENERATOR_SERVER_NODE)
    )

    @classmethod
    def prefix(cls, *args) -> Namespace:
        """
        Returns a Namespace object with the given arguments as the suffix.

        Args:
            *args: Variable length argument list.

        Returns:
            Namespace: A Namespace object with the given arguments as the suffix.
        """
        return Namespace("~configuration", *args)

    def __init__(self):
        # rospy.Subscriber(
        #     name=self.prefix(self.TOPIC_RECONFIGURE),
        #     data_class=std_msgs.Empty,
        #     callback=self.reconfigure
        # )
        ...


class TaskMode(Reconfigurable):
    _PROPS: Props_

    def __init__(self, props: Props_, **kwargs):
        Reconfigurable.__init__(self)
        self._PROPS = props


class Task(Props_):
    """
    Base class for defining tasks in the task generator module.

    Attributes:
        last_reset_time (int): The time of the last reset.

    Constants:
        TOPIC_RESET_START (str): The topic for the reset start event.
        TOPIC_RESET_END (str): The topic for the reset end event.
        PARAM_RESETTING (str): The parameter for resetting.

    Private Attributes:
        __reset_start (rospy.Publisher): The publisher for the reset start event.
        __reset_end (rospy.Publisher): The publisher for the reset end event.
        __reset_mutex (bool): The mutex for resetting.

    Args:
        obstacle_manager (ObstacleManager): The obstacle manager.
        robot_managers (List[RobotManager]): The list of robot managers.
        world_manager (WorldManager): The world manager.
        namespace (str, optional): The namespace for the task. Defaults to "".

    Raises:
        NotImplementedError: This class is meant to be subclassed and not instantiated directly.

    Methods:
        reset(**kwargs): Reset the task.
        is_done() -> bool: Check if the task is done.
        robot_names() -> List[str]: Get the names of the robots in the task.
        set_up_robot_managers(): Set up the robot managers.
        _clock_callback(clock: rosgraph_msgs.Clock): Callback function for the clock message.
        set_robot_position(position: PositionOrientation): Set the position of the robot.
        set_robot_goal(position: PositionOrientation): Set the goal position of the robot.
    """

    last_reset_time: int

    TOPIC_RESET_START = "reset_start"
    TOPIC_RESET_END = "reset_end"
    PARAM_RESETTING = "resetting"

    @classmethod
    def declare_parameters(cls, node: rclpy.node.Node):
        node.declare_parameter(cls.PARAM_RESETTING, True)


    __reset_start: rospy.Publisher
    __reset_end: rospy.Publisher
    __reset_mutex: bool

    def __init__(
        self,
        obstacle_manager: ObstacleManager,
        robot_managers: List[RobotManager],
        world_manager: WorldManager,
        namespace: str = "",
        *args,
        **kwargs
    ):
        raise NotImplementedError()

    def reset(self, **kwargs):
        ...

    @property
    def is_done(self) -> bool:
        return False

    @property
    def robot_names(self) -> List[str]:
        return [manager.name for manager in self.robot_managers]

    def set_up_robot_managers(self):
        for manager in self.robot_managers:
            manager.set_up_robot()

    def _clock_callback(self, clock: rosgraph_msgs.Clock):
        self.clock = clock

    def set_robot_position(self, position: PositionOrientation):
        ...

    def set_robot_goal(self, position: PositionOrientation):
        ...


from .task_factory import TaskFactory

def declare_modules():
    @TaskFactory.register_module(Constants.TaskMode.TM_Module.BENCHMARK)
    def _benchmark():
        from .modules.benchmark import Mod_Benchmark
        return Mod_Benchmark

    @TaskFactory.register_module(Constants.TaskMode.TM_Module.CLEAR_FORBIDDEN_ZONES)
    def _clear_forbidden_zones():
        from .modules.clear_forbidden_zones import Mod_ClearForbiddenZones
        return Mod_ClearForbiddenZones

    @TaskFactory.register_module(Constants.TaskMode.TM_Module.DYNAMIC_MAP)
    def _dynamic_map():
        from .modules.dynamic_map import Mod_DynamicMap
        return Mod_DynamicMap

    @TaskFactory.register_module(Constants.TaskMode.TM_Module.RVIZ_UI)
    def _rviz_ui():
        from .modules.rviz_ui import Mod_OverrideRobot
        return Mod_OverrideRobot

    @TaskFactory.register_module(Constants.TaskMode.TM_Module.STAGED)
    def _staged():
        from .modules.staged import Mod_Staged
        return Mod_Staged

def declare_obstacles():
    @TaskFactory.register_obstacles(Constants.TaskMode.TM_Obstacles.PARAMETRIZED)
    def _parametrized():
        from .obstacles.parametrized import TM_Parametrized
        return TM_Parametrized

    @TaskFactory.register_obstacles(Constants.TaskMode.TM_Obstacles.RANDOM)
    def _random():
        from .obstacles.random import TM_Random
        return TM_Random

    @TaskFactory.register_obstacles(Constants.TaskMode.TM_Obstacles.SCENARIO)
    def _scenario():
        from .obstacles.scenario import TM_Scenario
        return TM_Scenario
    
def declare_robots():
    @TaskFactory.register_robots(Constants.TaskMode.TM_Robots.EXPLORE)
    def _explore():
        from .robots.explore import TM_Explore
        return TM_Explore

    @TaskFactory.register_robots(Constants.TaskMode.TM_Robots.GUIDED)
    def _guided():
        from .robots.guided import TM_Guided
        return TM_Guided

    @TaskFactory.register_robots(Constants.TaskMode.TM_Robots.RANDOM)
    def _random():
        from .robots.random import TM_Random
        return TM_Random

    @TaskFactory.register_robots(Constants.TaskMode.TM_Robots.SCENARIO)
    def _scenario():
        from .robots.scenario import TM_Scenario
        return TM_Scenario
    
declare_modules()
declare_obstacles()
declare_robots()