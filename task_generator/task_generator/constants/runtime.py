import dataclasses
from typing import Any, Callable, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from task_generator.shared import Namespace

from . import Constants

@dataclasses.dataclass
class TaskConfig_General:
    WAIT_FOR_SERVICE_TIMEOUT: float = None
    MAX_RESET_FAIL_TIMES: int = None
    RNG: np.random.Generator = None
    DESIRED_EPISODES: float = None

@dataclasses.dataclass
class TaskConfig_Robot:
    GOAL_TOLERANCE_RADIUS: float = None
    GOAL_TOLERANCE_ANGLE: float = None
    SPAWN_ROBOT_SAFE_DIST: float = None
    TIMEOUT: float = None

@dataclasses.dataclass
class TaskConfig_Obstacles:
    OBSTACLE_MAX_RADIUS: float = None

@dataclasses.dataclass
class TaskConfig:
    General: TaskConfig_General = None
    Robot: TaskConfig_Robot = None
    Obstacles: TaskConfig_Obstacles = None

Config = TaskConfig()

class TaskGeneratorNode(Node):
    def __init__(self):
        super().__init__('task_generator_node')

        # Parameter deklarieren und auf Standardwerte setzen
        self.declare_parameter('timeout_wait_for_service', Constants.get_default("TIMEOUT_WAIT_FOR_SERVICE"))
        self.declare_parameter('max_reset_fail_times', Constants.get_default("MAX_RESET_FAIL_TIMES"))
        self.declare_parameter('goal_radius', Constants.get_default("GOAL_RADIUS"))
        self.declare_parameter('goal_tolerance_angle', Constants.get_default("GOAL_TOLERANCE_ANGLE"))
        self.declare_parameter('spawn_robot_safe_dist', Constants.get_default("SPAWN_ROBOT_SAFE_DIST"))
        self.declare_parameter('timeout', Constants.get_default("TIMEOUT"))
        self.declare_parameter('obstacle_max_radius', Constants.get_default("OBSTACLE_MAX_RADIUS"))
        self.declare_parameter('episodes', Constants.get_default("EPISODES"))

        # Konfiguration initialisieren
        global Config
        Config.General = TaskConfig_General(
            WAIT_FOR_SERVICE_TIMEOUT=self.get_parameter('timeout_wait_for_service').get_parameter_value().double_value,
            MAX_RESET_FAIL_TIMES=self.get_parameter('max_reset_fail_times').get_parameter_value().integer_value,
            RNG=np.random.default_rng(1),
            DESIRED_EPISODES=float(self.get_parameter('episodes').get_parameter_value().integer_value)
        )

        Config.Robot = TaskConfig_Robot(
            GOAL_TOLERANCE_RADIUS=self.get_parameter('goal_radius').get_parameter_value().double_value,
            GOAL_TOLERANCE_ANGLE=self.get_parameter('goal_tolerance_angle').get_parameter_value().double_value,
            SPAWN_ROBOT_SAFE_DIST=self.get_parameter('spawn_robot_safe_dist').get_parameter_value().double_value,
            TIMEOUT=self.get_parameter('timeout').get_parameter_value().double_value
        )

        Config.Obstacles = TaskConfig_Obstacles(
            OBSTACLE_MAX_RADIUS=self.get_parameter('obstacle_max_radius').get_parameter_value().double_value
        )

        # Parameter-Callback einrichten
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        global Config
        for param in params:
            if param.name == 'timeout_wait_for_service':
                Config.General.WAIT_FOR_SERVICE_TIMEOUT = param.value
            elif param.name == 'max_reset_fail_times':
                Config.General.MAX_RESET_FAIL_TIMES = param.value
            elif param.name == 'goal_radius':
                Config.Robot.GOAL_TOLERANCE_RADIUS = param.value
            elif param.name == 'goal_tolerance_angle':
                Config.Robot.GOAL_TOLERANCE_ANGLE = param.value
            elif param.name == 'spawn_robot_safe_dist':
                Config.Robot.SPAWN_ROBOT_SAFE_DIST = param.value
            elif param.name == 'timeout':
                Config.Robot.TIMEOUT = param.value
            elif param.name == 'obstacle_max_radius':
                Config.Obstacles.OBSTACLE_MAX_RADIUS = param.value
            elif param.name == 'episodes':
                Config.General.DESIRED_EPISODES = float(param.value)
        return rclpy.parameter.ParameterValue()

class FlatlandRandomModel:
    BODY = {
        "name": "base_link",
        "pose": [0, 0, 0],
        "color": [1, 0.2, 0.1, 1.0],
        "footprints": [],
    }
    FOOTPRINT = {
        "density": 1,
        "restitution": 1,
        "layers": ["all"],
        "collision": "true",
        "sensor": "false",
    }
    MIN_RADIUS = 0.2
    MAX_RADIUS = 0.6
    RANDOM_MOVE_PLUGIN = {
        "type": "RandomMove",
        "name": "RandomMove_Plugin",
        "body": "base_link",
    }
    LINEAR_VEL = 0.2
    ANGLUAR_VEL_MAX = 0.2

# no ~configuration possible because node is not fully initialized at this point
pedsim_ns = Namespace(
    "task_generator_node/configuration/pedsim/default_actor_config")

def lp(parameter: str, fallback: Any) -> Callable[[Optional[Any]], Any]:
    """
    load pedsim param
    """

    # load once at the start
    val = fallback

    gen = lambda: val

    if isinstance(val, list):
        lo, hi = val[:2]
        gen = lambda: min(
            hi,
            max(
                lo,
                Config.General.RNG.normal((hi + lo) / 2, (hi - lo) / 6)
            )
        )

    return lambda x: x if x is not None else gen()

class Pedsim:
    VMAX = lp("VMAX", 0.3)
    START_UP_MODE = lp("START_UP_MODE", "default")
    WAIT_TIME = lp("WAIT_TIME", 0.0)
    TRIGGER_ZONE_RADIUS = lp("TRIGGER_ZONE_RADIUS", 0.0)
    CHATTING_PROBABILITY = lp("CHATTING_PROBABILITY", 0.0)
    TELL_STORY_PROBABILITY = lp("TELL_STORY_PROBABILITY", 0.0)
    GROUP_TALKING_PROBABILITY = lp("GROUP_TALKING_PROBABILITY", 0.0)
    TALKING_AND_WALKING_PROBABILITY = lp("TALKING_AND_WALKING_PROBABILITY", 0.0)
    REQUESTING_SERVICE_PROBABILITY = lp("REQUESTING_SERVICE_PROBABILITY", 0.0)
    REQUESTING_GUIDE_PROBABILITY = lp("REQUESTING_GUIDE_PROBABILITY", 0.0)
    REQUESTING_FOLLOWER_PROBABILITY = lp("REQUESTING_FOLLOWER_PROBABILITY", 0.0)
    MAX_TALKING_DISTANCE = lp("MAX_TALKING_DISTANCE", 5.0)
    MAX_SERVICING_RADIUS = lp("MAX_SERVICING_RADIUS", 5.0)
    TALKING_BASE_TIME = lp("TALKING_BASE_TIME", 10.0)
    TELL_STORY_BASE_TIME = lp("TELL_STORY_BASE_TIME", 0.0)
    GROUP_TALKING_BASE_TIME = lp("GROUP_TALKING_BASE_TIME", 10.0)
    TALKING_AND_WALKING_BASE_TIME = lp("TALKING_AND_WALKING_BASE_TIME", 6.0)
    RECEIVING_SERVICE_BASE_TIME = lp("RECEIVING_SERVICE_BASE_TIME", 20.0)
    REQUESTING_SERVICE_BASE_TIME = lp("REQUESTING_SERVICE_BASE_TIME", 30.0)
    FORCE_FACTOR_DESIRED = lp("FORCE_FACTOR_DESIRED", 1.0)
    FORCE_FACTOR_OBSTACLE = lp("FORCE_FACTOR_OBSTACLE", 1.0)
    FORCE_FACTOR_SOCIAL = lp("FORCE_FACTOR_SOCIAL", 5.0)
    FORCE_FACTOR_ROBOT = lp("FORCE_FACTOR_ROBOT", 0.0)
    WAYPOINT_MODE = lp("WAYPOINT_MODE", 0)

def main(args=None):
    rclpy.init(args=args)
    node = TaskGeneratorNode()
    rclpy.spin(node)
    rclpy.shutdown()
