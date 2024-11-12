import dataclasses
import functools
import time
import os
import yaml
from typing import Callable, List, Collection, Dict, Any
from threading import Lock

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, Quaternion, PoseStamped
from hunav_msgs.srv import ComputeAgent, ComputeAgents, GetAgents, MoveAgent, ResetAgents
from hunav_msgs.msg import Agent, Agents, AgentBehavior
from std_srvs.srv import Empty, Trigger
from ament_index_python.packages import get_package_share_directory

from task_generator.constants import Constants
from task_generator.constants.runtime import Config
from task_generator.manager.entity_manager import EntityManager
from task_generator.manager.entity_manager.utils import (
    KnownObstacles,
    ObstacleLayer,
    SDFUtil,
    walls_to_obstacle,
)
from task_generator.shared import (
    DynamicObstacle,
    Model,
    ModelType,
    Namespace,
    Obstacle,
    PositionOrientation,
    Robot,
)
from task_generator.simulators.gazebo_simulator import GazeboSimulator
from task_generator.utils.geometry import quaternion_from_euler, euler_from_quaternion


    
class HunavsimManager(EntityManager):
    # Class constants
    WALLS_ENTITY = "walls"  # Definition for walls_entity
    
    # Animation configuration (from WorldGenerator)
    SKIN_TYPES = {
        0: 'elegant_man.dae',
        1: 'casual_man.dae',
        2: 'elegant_woman.dae',
        3: 'regular_man.dae',
        4: 'worker_man.dae',
        5: 'walk.dae'
    }
    

    ANIMATION_TYPES = {
        'WALK': '07_01-walk.bvh',
        'WALK_FORWARD': '69_02_walk_forward.bvh',
        'NORMAL_WAIT': '137_28-normal_wait.bvh',
        'WALK_CHILDISH': '142_01-walk_childist.bvh',
        'SLOW_WALK': '07_04-slow_walk.bvh',
        'WALK_SCARED': '142_17-walk_scared.bvh',
        'WALK_ANGRY': '17_01-walk_with_anger.bvh'
    }
    # Service Names
    SERVICE_COMPUTE_AGENT = 'compute_agent'
    SERVICE_COMPUTE_AGENTS = 'compute_agents'
    SERVICE_GET_AGENTS = 'get_agents'
    SERVICE_MOVE_AGENT = 'move_agent'
    SERVICE_RESET_AGENTS = 'reset_agents'
    
    def __init__(self, namespace: Namespace, simulator: GazeboSimulator, node: Node = None):
        if node is None:
            from task_generator import TASKGEN_NODE
            node = TASKGEN_NODE
            
        super().__init__(namespace=namespace, simulator=simulator, node=node)
        
        # Initialize state variables
        self._is_paused = False
        self._semaphore_reset = False
        self._agents_initialized = False
        self._robot_initialized = False
        self._lock = Lock()
        self._update_rate = 0.1
        
        # Initialize collections
        self._known_obstacles = KnownObstacles()  # Initialization
        self._pedestrians = {}  # Store pedestrian states
        
        # Setup services
        self.setup_services()
        
        # Setup timer for pedestrian updates
        self._update_timer = self._node.create_timer(
            self._update_rate,
            self.update_pedestrians
        )
        
        # Initialize JAIL_POS generator
        def gen_JAIL_POS(steps: int, x: int = 1, y: int = 0):
            steps = max(steps, 1)
            while True:
                x += y == steps
                y %= steps
                yield PositionOrientation(-x, y, 0)
                y += 1
        self.JAIL_POS = gen_JAIL_POS(10)

    # def load_agent_configurations(self):                                          ##Currently not needed anymore since Agentparameter are integrated in the Dynamic Obstacle class itself (shared.py)
    #     """Load agent configurations from YAML (WorldGenerator functionality)"""
    #     config_path = os.path.join(
    #         get_package_share_directory('arena_bringup'),
    #         'configs',
    #         'hunav_agents',
    #         'default.yaml'
    #     )
    #     self._node.get_logger().info(f"Loading config from: {config_path}")

    #     with open(config_path, 'r') as f:
    #         self.agent_config = yaml.safe_load(f)['hunav_loader']['ros__parameters'] 
    #         self._node.get_logger().info("Loaded agent configurations:") # DEBUG TERMINAL OUTPUT
    #         self._node.get_logger().info(f"{yaml.dump(self.agent_config, indent=2)}") #DEBUG TERMINAL OUTPUT 
            
    #     self._known_obstacles = KnownObstacles()
    #     self._pedestrians = {}  # Store pedestrian states
        
    #     # Process configurations like WorldGenerator
    #     for agent_name, config in self.agent_config.items():
    #         if isinstance(config, dict) and 'id' in config:
    #             self._pedestrians[agent_name] = {
    #                 'config': config,
    #                 'current_animation': 'WALK',
    #                 'animation_time': 0.0,
    #                 'last_update': time.time()
    #             }

    def setup_services(self):
        """Initialize all required services"""
        self._node.get_logger().info("Setting up Hunavservices...")

        # First create Service-Clients 
        self._compute_agent_client = self._node.create_client(
            ComputeAgent,
            self._namespace(self.SERVICE_COMPUTE_AGENT)
        )
        self._compute_agents_client = self._node.create_client(
            ComputeAgents,
            self._namespace(self.SERVICE_COMPUTE_AGENTS)
        )
        self._get_agents_client = self._node.create_client(
            GetAgents,
            self._namespace(self.SERVICE_GET_AGENTS)
        )
        self._move_agent_client = self._node.create_client(
            MoveAgent,
            self._namespace(self.SERVICE_MOVE_AGENT)
        )
        self._reset_agents_client = self._node.create_client(
            ResetAgents,
            self._namespace(self.SERVICE_RESET_AGENTS)
        )

        # Wait for Services 
        required_services = [
            (self._compute_agent_client, 'compute_agent'),
            (self._compute_agents_client, 'compute_agents'),
            (self._get_agents_client, 'get_agents'),
            (self._move_agent_client, 'move_agent'),
            (self._reset_agents_client, 'reset_agents')
        ]
        
        max_attempts = 5
        for client, name in required_services:
            attempts = 0
            while attempts < max_attempts:
                if client.wait_for_service(timeout_sec=2.0):
                    self._node.get_logger().info(f'Service {name} is available')
                    break
                attempts += 1
                self._node.get_logger().warn(f'Waiting for service {name} (attempt {attempts}/{max_attempts})')
                
            if attempts >= max_attempts:
                self._node.get_logger().error(f'Service {name} not available after {max_attempts} attempts')
                return False
                
        self._node.get_logger().info("All services are ready")
        return True

    def create_pedestrian_sdf(self, agent_config: Dict) -> str:
        """Create SDF description for pedestrian (from WorldGenerator)"""
        skin_type = self.SKIN_TYPES.get(agent_config.get('skin', 0), 'casual_man.dae')
        
        # Get path to mesh file
        mesh_path = os.path.join(
            get_package_share_directory('hunav_sim'),
            'hunav_rviz2_panel/meshes',
            skin_type
        )
        
        # Height adjustment based on skin type (from HuNavPlugin)
        height_adjustments = {
            'elegant_man.dae': 0.96,
            'casual_man.dae': 0.97,
            'elegant_woman.dae': 0.93,
            'regular_man.dae': 0.93,
            'worker_man.dae': 0.97
        }
        z_pos = height_adjustments.get(skin_type, 1.0)
        
        sdf = f"""<?xml version="1.0" ?>
        <sdf version="1.6">
            <model name="{agent_config['name']}">
                <static>false</static>
                <pose>0 0 {z_pos} 0 0 0</pose>
                <link name="link">
                    <visual name="visual">
                        <geometry>
                            <mesh>
                                <uri>file://{mesh_path}</uri>
                            </mesh>
                        </geometry>
                    </visual>
                    <collision name="collision">
                        <geometry>
                            <cylinder>
                                <radius>0.3</radius>
                                <length>1.7</length>
                            </cylinder>
                        </geometry>
                    </collision>
                </link>
            </model>
        </sdf>"""
        return sdf

    def spawn_dynamic_obstacles(self, obstacles: Collection[DynamicObstacle]):
        """Spawn dynamic obstacles/agents with enhanced debug output"""
        # Initial debug prints
        print("\n==================== STARTING SPAWN PROCESS ====================")
        print(f"Attempting to spawn {len(list(obstacles))} obstacles")
        self._node.get_logger().error(f"Attempting to spawn {len(list(obstacles))} obstacles")

        for obstacle in obstacles:
            print("\n=============== NEW OBSTACLE PROCESSING ===============")
            print(f"Processing obstacle: {obstacle.name}")
            self._node.get_logger().error(f"Processing obstacle: {obstacle.name}")

            # Create Hunav Agent
            request = ComputeAgent.Request()
            agent = Agent()

            try:
                # Basic Properties Debug
                print("\n--- Basic Properties Debug ---")
                print(f"ID: {obstacle.id}")
                print(f"Type: {obstacle.type}")
                print(f"Skin: {obstacle.skin}")
                print(f"Name: {obstacle.name}")
                print(f"Group ID: {obstacle.group_id}")
                self._node.get_logger().error(f"Basic Properties - ID: {obstacle.id}, Type: {obstacle.type}, Skin: {obstacle.skin}")

                # Set basic properties
                agent.id = obstacle.id
                agent.type = obstacle.type
                agent.skin = obstacle.skin
                agent.name = obstacle.name
                agent.group_id = obstacle.group_id
            except Exception as e:
                print(f"ERROR in basic properties: {e}")
                self._node.get_logger().error(f"ERROR in basic properties: {e}")

            try:
                # Position Debug
                print("\n--- Position & Orientation Debug ---")
                print(f"Position object: {obstacle.position}")
                print(f"Position type: {type(obstacle.position)}")
                print(f"Yaw value: {obstacle.yaw}")
                self._node.get_logger().error(f"Position Data: {obstacle.position}, Yaw: {obstacle.yaw}")

                # Set position
                agent.position = obstacle.position
                agent.yaw = obstacle.yaw
            except Exception as e:
                print(f"ERROR in position setting: {e}")
                self._node.get_logger().error(f"ERROR in position setting: {e}")

            try:
                # Velocity Debug
                print("\n--- Velocity Debug ---")
                print(f"Velocity object: {obstacle.velocity}")
                print(f"Desired velocity: {obstacle.desired_velocity}")
                print(f"Linear vel: {obstacle.linear_vel}")
                print(f"Angular vel: {obstacle.angular_vel}")
                print(f"Radius: {obstacle.radius}")
                self._node.get_logger().error(f"Velocity Data - Desired: {obstacle.desired_velocity}, Linear: {obstacle.linear_vel}")

                # Set velocity
                agent.velocity = obstacle.velocity if obstacle.velocity else Twist()
                agent.desired_velocity = obstacle.desired_velocity
                agent.radius = obstacle.radius
                agent.linear_vel = obstacle.linear_vel
                agent.angular_vel = obstacle.angular_vel
            except Exception as e:
                print(f"ERROR in velocity setting: {e}")
                self._node.get_logger().error(f"ERROR in velocity setting: {e}")

            try:
                # Behavior Debug
                print("\n--- Behavior Debug ---")
                print(f"Behavior type: {obstacle.behavior.type}")
                print(f"Configuration: {obstacle.behavior.configuration}")
                print(f"Duration: {obstacle.behavior.duration}")
                print("Force Factors:")
                print(f"- Goal: {obstacle.behavior.goal_force_factor}")
                print(f"- Obstacle: {obstacle.behavior.obstacle_force_factor}")
                print(f"- Social: {obstacle.behavior.social_force_factor}")
                print(f"- Other: {obstacle.behavior.other_force_factor}")
                self._node.get_logger().error(f"Behavior Data - Type: {obstacle.behavior.type}, Config: {obstacle.behavior.configuration}")

                # Set behavior
                agent.behavior = AgentBehavior()
                agent.behavior.type = obstacle.behavior.type
                agent.behavior.configuration = obstacle.behavior.configuration
                agent.behavior.duration = obstacle.behavior.duration
                agent.behavior.once = obstacle.behavior.once
                agent.behavior.vel = obstacle.behavior.vel
                agent.behavior.dist = obstacle.behavior.dist
                agent.behavior.goal_force_factor = obstacle.behavior.goal_force_factor
                agent.behavior.obstacle_force_factor = obstacle.behavior.obstacle_force_factor
                agent.behavior.social_force_factor = obstacle.behavior.social_force_factor
                agent.behavior.other_force_factor = obstacle.behavior.other_force_factor
            except Exception as e:
                print(f"ERROR in behavior setting: {e}")
                self._node.get_logger().error(f"ERROR in behavior setting: {e}")

            try:
                # Goals Debug
                print("\n--- Goals Debug ---")
                print(f"Number of goals: {len(obstacle.goals)}")
                for i, goal in enumerate(obstacle.goals):
                    print(f"Goal {i}: {goal}")
                print(f"Cyclic goals: {obstacle.cyclic_goals}")
                print(f"Goal radius: {obstacle.goal_radius}")
                self._node.get_logger().error(f"Goals Data - Count: {len(obstacle.goals)}, Cyclic: {obstacle.cyclic_goals}")

                # Set goals
                agent.goals = obstacle.goals
                agent.cyclic_goals = obstacle.cyclic_goals
                agent.goal_radius = obstacle.goal_radius
            except Exception as e:
                print(f"ERROR in goals setting: {e}")
                self._node.get_logger().error(f"ERROR in goals setting: {e}")

            try:
                # Closest obstacles Debug
                print("\n--- Closest Obstacles Debug ---")
                print(f"Number of closest obstacles: {len(obstacle.closest_obs)}")
                self._node.get_logger().error(f"Closest obstacles count: {len(obstacle.closest_obs)}")

                agent.closest_obs = obstacle.closest_obs
            except Exception as e:
                print(f"ERROR in closest obstacles setting: {e}")
                self._node.get_logger().error(f"ERROR in closest obstacles setting: {e}")

            try:
                # SDF Model Creation Debug
                print("\n--- SDF Model Creation ---")
                print("Creating SDF model...")
                self._node.get_logger().error("Starting SDF model creation")
                
                sdf = self.create_pedestrian_sdf(obstacle)
                print("SDF model created successfully")
                
                # Create model with SDF
                obstacle = dataclasses.replace(
                    obstacle,
                    model=Model(
                        description=sdf,
                        model_type=ModelType.SDF,
                        name=obstacle.name
                    )
                )
            except Exception as e:
                print(f"ERROR in SDF model creation: {e}")
                self._node.get_logger().error(f"ERROR in SDF model creation: {e}")

            try:
                # Spawn Entity Debug
                print("\n--- Entity Spawning ---")
                print(f"Attempting to spawn entity: {obstacle.name}")
                spawn_success = self._simulator.spawn_entity(obstacle)
                print(f"Spawn {'successful' if spawn_success else 'failed'}")
                self._node.get_logger().error(f"Spawn result for {obstacle.name}: {'success' if spawn_success else 'failed'}")
            except Exception as e:
                print(f"ERROR in entity spawning: {e}")
                self._node.get_logger().error(f"ERROR in entity spawning: {e}")

            try:
                # HuNav Registration Debug
                print("\n--- HuNav Registration ---")
                print("Registering with HuNav...")
                request.agent = agent
                future = self._compute_agent_client.call_async(request)
                rclpy.spin_until_future_complete(self._node, future)
                
                if future.result():
                    print("Successfully registered with HuNav")
                    self._node.get_logger().error("Successfully registered with HuNav")
                else:
                    print("Failed to register with HuNav")
                    self._node.get_logger().error("Failed to register with HuNav")
            except Exception as e:
                print(f"ERROR in HuNav registration: {e}")
                self._node.get_logger().error(f"ERROR in HuNav registration: {e}")

            # Final steps
            known = self._known_obstacles.create_or_get(
                name=obstacle.name,
                obstacle=obstacle,
                hunav_spawned=True,
                layer=ObstacleLayer.INUSE,
            )
            
            print("\n=============== OBSTACLE PROCESSING COMPLETE ===============")
            self._node.get_logger().error("OBSTACLE PROCESSING COMPLETE")

        print("\n==================== SPAWN PROCESS COMPLETE ====================")
        self._node.get_logger().error("SPAWN PROCESS COMPLETE")

    def update_pedestrians(self):
        """Update pedestrians (from HuNavPlugin's OnUpdate)"""
        with self._lock:
            current_time = time.time()
            
            # Get updates from Hunav
            request = ComputeAgents.Request()
            future = self._compute_agents_client.call_async(request)
            rclpy.spin_until_future_complete(self._node, future)
            
            if future.result() is not None:
                agents = future.result().agents
                for agent in agents:
                    if agent.id in self._pedestrians:
                        ped_data = self._pedestrians[agent.id]
                        
                        # Update position
                        pos = PositionOrientation(
                            x=agent.position.position.x,
                            y=agent.position.position.y,
                            orientation=agent.yaw
                        )
                        self._simulator.move_entity(name=agent.id, position=pos)
                        
                        # Update animation (like HuNavPlugin)
                        dt = current_time - ped_data['last_update']
                        animation_factor = self.get_animation_factor(agent.behavior)
                        ped_data['animation_time'] += dt * animation_factor
                        
                        # Update animation state if needed
                        if agent.behavior.state != ped_data.get('current_state'):
                            self.update_agent_animation(agent.id, agent.behavior)
                            ped_data['current_state'] = agent.behavior.state
                        
                        ped_data['last_update'] = current_time

    def get_animation_factor(self, behavior: AgentBehavior) -> float:
        """Get animation speed factor based on behavior (from HuNavPlugin)"""
        if behavior.state == AgentBehavior.BEH_NO_ACTIVE:
            return 1.0
            
        if behavior.type == AgentBehavior.BEH_REGULAR:
            return 1.5
        elif behavior.type == AgentBehavior.BEH_IMPASSIVE:
            return 1.5
        elif behavior.type == AgentBehavior.BEH_SURPRISED:
            return 1.0
        elif behavior.type == AgentBehavior.BEH_THREATENING:
            return 1.0
        elif behavior.type == AgentBehavior.BEH_SCARED:
            return 1.5
        elif behavior.type == AgentBehavior.BEH_CURIOUS:
            return 1.0
        
        return 1.0

    def update_agent_animation(self, agent_id: str, behavior: AgentBehavior):
        """Update agent animation based on behavior (from HuNavPlugin)"""
        if behavior.state == AgentBehavior.BEH_NO_ACTIVE:
            animation = 'NORMAL_WAIT'
        else:
            if behavior.type == AgentBehavior.BEH_REGULAR:
                animation = 'WALK'
            elif behavior.type == AgentBehavior.BEH_IMPASSIVE:
                animation = 'WALK_FORWARD'
            elif behavior.type == AgentBehavior.BEH_SURPRISED:
                animation = 'NORMAL_WAIT'
            elif behavior.type == AgentBehavior.BEH_THREATENING:
                animation = 'WALK_ANGRY'
            elif behavior.type == AgentBehavior.BEH_SCARED:
                animation = 'WALK_SCARED'
            elif behavior.type == AgentBehavior.BEH_CURIOUS:
                animation = 'SLOW_WALK'
            else:
                animation = 'WALK'
                
        if agent_id in self._pedestrians:
            self._pedestrians[agent_id]['current_animation'] = animation


    def _load_agent_config(self, agent_id: str) -> Dict[str, Any]:
            """Load configuration for a specific agent
            
            Args:
                agent_id (str): ID of the agent
                
            Returns:
                Dict[str, Any]: Configuration dictionary
            """
            if agent_id not in self.agent_config:
                self._node.get_logger().warn(f"No configuration found for agent {agent_id}")
                return {}
            return self.agent_config[agent_id]

    def spawn_obstacles(self, obstacles):
        """Spawn static obstacles"""
        for obstacle in obstacles:
            known = self._known_obstacles.get(obstacle.name)
            if known is not None:
                if known.obstacle.name != obstacle.name:
                    raise RuntimeError(
                        f"New model name {obstacle.name} does not match model name {known.obstacle.name} of known obstacle {obstacle.name}"
                    )
                known.layer = ObstacleLayer.INUSE
            else:
                known = self._known_obstacles.create_or_get(
                    name=obstacle.name,
                    obstacle=obstacle,
                    hunav_spawned=False,
                    layer=ObstacleLayer.INUSE,
                )
            
            self._simulator.spawn_entity(obstacle)

    def spawn_walls(self, walls, heightmap):
        """Spawn walls in simulation"""
        if self.WALLS_ENTITY in self._known_obstacles:
            return

        obstacle = walls_to_obstacle(heightmap)
        self._known_obstacles.create_or_get(
            name=self.WALLS_ENTITY,
            obstacle=obstacle,
            layer=ObstacleLayer.WORLD,
            hunav_spawned=False,
        )
        
        self._simulator.spawn_entity(obstacle)

    def remove_obstacles(self, purge):
        """Remove obstacles based on purge level"""
        if not self._is_paused:
            self._is_paused = True
            request = ResetAgents.Request()
            future = self._reset_agents_client.call_async(request)
            rclpy.spin_until_future_complete(self._node, future)

        while self._semaphore_reset:
            time.sleep(0.1)

        actions: List[Callable] = []
        self._semaphore_reset = True

        try:
            to_forget: List[str] = list()

            for obstacle_id, obstacle in self._known_obstacles.items():
                if purge >= obstacle.layer:
                    if isinstance(self._simulator, GazeboSimulator):
                        if isinstance(obstacle.obstacle, DynamicObstacle):
                            def move_to_jail(obstacle_id):
                                jail = next(self.JAIL_POS)
                                self._simulator.move_entity(name=obstacle_id, position=jail)
                            actions.append(functools.partial(move_to_jail, obstacle_id))
                        else:
                            def delete_entity(obstacle_id):
                                obstacle.hunav_spawned = False
                                self._simulator.delete_entity(name=obstacle_id)
                            actions.append(functools.partial(delete_entity, obstacle_id))
                            to_forget.append(obstacle_id)
                    else:
                        obstacle.hunav_spawned = False
                        to_forget.append(obstacle_id)

            for obstacle_id in to_forget:
                self._known_obstacles.forget(name=obstacle_id)
                if obstacle_id in self._pedestrians:
                    del self._pedestrians[obstacle_id]

        finally:
            self._semaphore_reset = False

        for action in actions:
            action()

    def move_robot(self, name: str, position: PositionOrientation):
        """Move robot to new position"""
        request = MoveAgent.Request()
        request.id = name
        request.pose.position = Point(x=position.x, y=position.y, z=0.0)
        quat = quaternion_from_euler(0.0, 0.0, position.orientation)
        request.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        
        future = self._move_agent_client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future)

    def spawn_robot(self, robot: Robot):
        """Spawn robot in simulation"""
        self._simulator.spawn_entity(robot)

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] (from HuNavPlugin)"""
        while angle <= -3.14159:
            angle += 2 * 3.14159
        while angle > 3.14159:
            angle -= 2 * 3.14159
        return angle