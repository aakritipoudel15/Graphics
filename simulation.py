import numpy as np
import pygame
import random
from dataclasses import dataclass
from typing import List, Tuple, Set

@dataclass
class Agent:
    position: np.ndarray
    velocity: np.ndarray
    pheromone_list: List[Tuple[float, float, float]]  # (x, y, strength)
    
class HybridSwarmSimulation:
    def __init__(self, width=800, height=600, n_agents=20, n_targets=8, n_obstacles=4):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Hybrid Swarm Simulation")
        
        # Simulation parameters
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.n_obstacles = n_obstacles
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
        # ACO/PSO parameters
        self.psi_e = 1.0  # exploration weight
        self.psi_c = 0.5  # cognitive weight
        self.psi_s = 0.3  # social weight
        self.evaporation_rate = 0.1
        self.max_velocity = 5.0  # Added missing parameter
        
        # Initialize simulation components
        self.agents = self._initialize_agents()
        self.targets = self._initialize_targets()
        self.obstacles = self._initialize_obstacles()
        
    def _initialize_agents(self) -> List[Agent]:
        agents = []
        for _ in range(self.n_agents):
            pos = np.array([random.uniform(0, self.width),
                          random.uniform(0, self.height)])
            vel = np.array([random.uniform(-self.max_velocity, self.max_velocity),
                          random.uniform(-self.max_velocity, self.max_velocity)])
            agents.append(Agent(pos, vel, []))
        return agents
    
    def _initialize_targets(self) -> Set[Tuple[float, float]]:
        targets = set()
        while len(targets) < self.n_targets:
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            targets.add((x, y))
        return targets
    
    def _initialize_obstacles(self) -> List[pygame.Rect]:
        obstacles = []
        for _ in range(self.n_obstacles):
            x = random.uniform(0, self.width - 50)
            y = random.uniform(0, self.height - 100)
            w = random.uniform(20, 50)
            h = random.uniform(50, 100)
            obstacles.append(pygame.Rect(x, y, w, h))
        return obstacles
    
    def update_agent_position(self, agent: Agent):
        # Update velocity using equation (6) from the algorithm
        exploration = self.psi_e * np.random.rand(2) * 2 - 1
        cognitive = self.psi_c * np.random.rand(2) * 2 - 1
        social = np.zeros(2)
        
        if agent.pheromone_list:
            # Calculate social component based on pheromones
            pheromone_positions = np.array([p[:2] for p in agent.pheromone_list])
            strengths = np.array([p[2] for p in agent.pheromone_list])
            weighted_direction = np.average(pheromone_positions - agent.position, 
                                         weights=strengths, axis=0)
            social = self.psi_s * weighted_direction
        
        agent.velocity = agent.velocity + exploration + cognitive + social
        
        # Limit velocity
        speed = np.linalg.norm(agent.velocity)
        if speed > self.max_velocity:
            agent.velocity = (agent.velocity / speed) * self.max_velocity
        
        # Update position using equation (7)
        agent.position = agent.position + agent.velocity
        
        # Boundary conditions
        agent.position[0] = np.clip(agent.position[0], 0, self.width)
        agent.position[1] = np.clip(agent.position[1], 0, self.height)
    
    def update(self):
        # Update all agents
        for agent in self.agents:
            self.update_agent_position(agent)
            
            # Check for targets
            for target in list(self.targets):
                distance = np.linalg.norm(agent.position - np.array(target))
                if distance < 20:  # Detection radius
                    self.targets.remove(target)
                    # Add pheromone at target location
                    agent.pheromone_list.append((*target, 1.0))
        
        # Update pheromones
        for agent in self.agents:
            for i, (x, y, strength) in enumerate(agent.pheromone_list):
                agent.pheromone_list[i] = (x, y, strength * (1 - self.evaporation_rate))
            # Remove weak pheromones
            agent.pheromone_list = [(x, y, s) for x, y, s in agent.pheromone_list if s > 0.1]
    
    def render(self):
        self.screen.fill(self.WHITE)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.GRAY, obstacle)
        
        # Draw targets
        for target in self.targets:
            pygame.draw.circle(self.screen, self.RED, 
                             (int(target[0]), int(target[1])), 10)
        
        # Draw agents and their connections
        for agent in self.agents:
            # Draw agent
            pygame.draw.circle(self.screen, self.BLUE,
                             (int(agent.position[0]), int(agent.position[1])), 5)
            
            # Draw pheromone connections
            for x, y, strength in agent.pheromone_list:
                color = (0, int(255 * strength), int(255 * strength))
                pygame.draw.line(self.screen, color,
                               (int(agent.position[0]), int(agent.position[1])),
                               (int(x), int(y)), 1)
        
        pygame.display.flip()
    
    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            self.update()
            self.render()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    simulation = HybridSwarmSimulation()
    simulation.run()

# import numpy as np
# import pygame
# import random
# from dataclasses import dataclass
# from typing import List, Tuple, Set

# @dataclass
# class Agent:
#     position: np.ndarray
#     velocity: np.ndarray
#     pheromone_list: List[Tuple[float, float, float]]  # (x, y, strength)
#     discovered_target: bool = False  # Track if agent found a target

# class HybridSwarmSimulation:
#     def __init__(self, width=800, height=600, n_agents=20, n_targets=8, n_obstacles=4):
#         # Initialize Pygame
#         pygame.init()
#         self.screen = pygame.display.set_mode((width, height))
#         pygame.display.set_caption("Hybrid Swarm Simulation")
        
#         # Simulation parameters
#         self.width = width
#         self.height = height
#         self.n_agents = n_agents
#         self.n_targets = n_targets
#         self.n_obstacles = n_obstacles
        
#         # Colors
#         self.WHITE = (255, 255, 255)
#         self.BLACK = (0, 0, 0)
#         self.RED = (255, 0, 0)
#         self.BLUE = (0, 0, 255)
#         self.GRAY = (128, 128, 128)
#         self.GREEN = (0, 255, 0)
        
#         # ACO/PSO parameters
#         self.psi_e = 1.0  # exploration weight
#         self.psi_c = 0.5  # cognitive weight
#         self.psi_s = 0.3  # social weight
#         self.evaporation_rate = 0.1
#         self.max_velocity = 5.0
        
#         # Initialize simulation components
#         self.agents = self._initialize_agents()
#         self.targets = self._initialize_targets()
#         self.obstacles = self._initialize_obstacles()
        
#     def _initialize_agents(self) -> List[Agent]:
#         agents = []
#         for _ in range(self.n_agents):
#             pos = np.array([random.uniform(0, self.width), random.uniform(0, self.height)])
#             vel = np.array([random.uniform(-self.max_velocity, self.max_velocity),
#                             random.uniform(-self.max_velocity, self.max_velocity)])
#             agents.append(Agent(pos, vel, []))
#         return agents
    
#     def _initialize_targets(self) -> Set[Tuple[float, float]]:
#         targets = set()
#         while len(targets) < self.n_targets:
#             x = random.uniform(50, self.width - 50)
#             y = random.uniform(50, self.height - 50)
#             targets.add((x, y))
#         return targets
    
#     def _initialize_obstacles(self) -> List[pygame.Rect]:
#         obstacles = []
#         for _ in range(self.n_obstacles):
#             x = random.uniform(0, self.width - 50)
#             y = random.uniform(0, self.height - 100)
#             w = random.uniform(20, 50)
#             h = random.uniform(50, 100)
#             obstacles.append(pygame.Rect(x, y, w, h))
#         return obstacles
    
#     def update_agent_position(self, agent: Agent):
#         # Update velocity using hybrid ACO/PSO logic
#         exploration = self.psi_e * (np.random.rand(2) * 2 - 1)
#         cognitive = self.psi_c * (np.random.rand(2) * 2 - 1)
#         social = np.zeros(2)
        
#         if agent.pheromone_list:
#             # Calculate social component based on pheromones
#             pheromone_positions = np.array([p[:2] for p in agent.pheromone_list])
#             strengths = np.array([p[2] for p in agent.pheromone_list])
#             weighted_directions = pheromone_positions - agent.position
#             normalized_weights = strengths / np.sum(strengths)
#             weighted_direction = np.sum(normalized_weights[:, None] * weighted_directions, axis=0)
#             social = self.psi_s * weighted_direction
        
#         # Obstacle avoidance
#         repulsion = np.zeros(2)
#         for obstacle in self.obstacles:
#             obstacle_center = np.array([obstacle.x + obstacle.width / 2, obstacle.y + obstacle.height / 2])
#             distance = np.linalg.norm(agent.position - obstacle_center)
#             if distance < 50:  # Repulsion range
#                 repulsion += 50 / distance * (agent.position - obstacle_center)
        
#         agent.velocity += exploration + cognitive + social + repulsion
        
#         # Limit velocity
#         speed = np.linalg.norm(agent.velocity)
#         if speed > self.max_velocity:
#             agent.velocity = (agent.velocity / speed) * self.max_velocity
        
#         # Update position
#         agent.position += agent.velocity
        
#         # Boundary conditions
#         agent.position[0] = np.clip(agent.position[0], 5, self.width - 5)
#         agent.position[1] = np.clip(agent.position[1], 5, self.height - 5)
    
#     def update(self):
#         # Update all agents
#         for agent in self.agents:
#             self.update_agent_position(agent)
            
#             # Check for targets
#             for target in list(self.targets):
#                 distance = np.linalg.norm(agent.position - np.array(target))
#                 if distance < 30:  # Detection radius
#                     self.targets.remove(target)
#                     agent.pheromone_list.append((*target, 1.0))  # Add pheromone
#                     agent.discovered_target = True
        
#         # Update pheromones
#         for agent in self.agents:
#             for i, (x, y, strength) in enumerate(agent.pheromone_list):
#                 agent.pheromone_list[i] = (x, y, strength * (1 - self.evaporation_rate))
#             # Remove weak pheromones
#             agent.pheromone_list = [(x, y, s) for x, y, s in agent.pheromone_list if s > 0.1]
    
#     def render(self):
#         self.screen.fill(self.WHITE)
        
#         # Draw obstacles
#         for obstacle in self.obstacles:
#             pygame.draw.rect(self.screen, self.GRAY, obstacle)
        
#         # Draw targets
#         for target in self.targets:
#             pygame.draw.circle(self.screen, self.RED, (int(target[0]), int(target[1])), 10)
        
#         # Draw agents
#         for agent in self.agents:
#             color = self.GREEN if agent.discovered_target else self.BLUE
#             pygame.draw.circle(self.screen, color, (int(agent.position[0]), int(agent.position[1])), 5)
#             for x, y, strength in agent.pheromone_list:
#                 color = (0, int(255 * strength), int(255 * strength))
#                 pygame.draw.line(self.screen, color,
#                                  (int(agent.position[0]), int(agent.position[1])),
#                                  (int(x), int(y)), 1)
        
#         pygame.display.flip()
    
#     def run(self):
#         running = True
#         clock = pygame.time.Clock()
        
#         while running:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
            
#             self.update()
#             self.render()
#             clock.tick(60)
        
#         pygame.quit()

# if __name__ == "__main__":
#     simulation = HybridSwarmSimulation()
#     simulation.run()
