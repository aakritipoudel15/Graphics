# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np

# # Parameters
# width, height = 100, 100
# n_agents = 10
# n_targets = 5
# agent_radius = 1.0
# target_radius = 2.0
# max_velocity = 2.0

# # Initialize agents and targets
# agents = np.random.rand(n_agents, 2) * [width, height]  # Random positions
# velocities = (np.random.rand(n_agents, 2) * 2 - 1) * max_velocity  # Random velocities
# targets = np.random.rand(n_targets, 2) * [width, height]  # Random target positions

# # Animation setup
# fig, ax = plt.subplots()
# ax.set_xlim(0, width)
# ax.set_ylim(0, height)
# agent_dots, = ax.plot([], [], 'bo', markersize=8)  # Blue agents
# target_dots, = ax.plot([], [], 'ro', markersize=12)  # Red targets

# def init():
#     """Initialize the animation."""
#     agent_dots.set_data([], [])
#     target_dots.set_data(targets[:, 0], targets[:, 1])
#     return agent_dots, target_dots

# def update(frame):
#     """Update agent positions."""
#     global agents, velocities, targets
    
#     # Move agents
#     agents += velocities
    
#     # Reflect agents off the boundaries
#     for i in range(n_agents):
#         if agents[i, 0] < 0 or agents[i, 0] > width:
#             velocities[i, 0] *= -1
#         if agents[i, 1] < 0 or agents[i, 1] > height:
#             velocities[i, 1] *= -1
    
#     # Check if agents reach targets
#     for i, target in enumerate(targets):
#         distances = np.linalg.norm(agents - target, axis=1)
#         reached = distances < (agent_radius + target_radius)
#         if np.any(reached):
#             targets[i] = np.random.rand(2) * [width, height]  # Relocate target

#     # Update the positions
#     agent_dots.set_data(agents[:, 0], agents[:, 1])
#     target_dots.set_data(targets[:, 0], targets[:, 1])
#     return agent_dots, target_dots

# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=50)
# plt.show()


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Simulation Parameters
width, height = 100, 100
n_agents = 20
n_targets = 5
n_obstacles = 3
agent_radius = 1.0
target_radius = 2.0
comm_radius = 10.0
visibility_radius = 5.0
pheromone_decay = 0.95
pheromone_intensity = 5.0
max_velocity = 1.0
frames = 300

# Initialize agents, targets, and obstacles
agents = np.random.rand(n_agents, 2) * [width, height]
velocities = (np.random.rand(n_agents, 2) * 2 - 1) * max_velocity
targets = np.random.rand(n_targets, 2) * [width, height]
obstacles = np.random.rand(n_obstacles, 2) * [width, height]
pheromone_grid = np.zeros((width, height))  # 2D grid for pheromone trails

# Animation setup
fig, ax = plt.subplots()
ax.set_xlim(0, width)
ax.set_ylim(0, height)

# Plot elements
agent_dots, = ax.plot([], [], 'bo', markersize=8, label='Agents')
target_dots, = ax.plot([], [], 'ro', markersize=10, label='Targets')
obstacle_dots, = ax.plot([], [], 'ks', markersize=12, label='Obstacles')
pheromone_img = ax.imshow(pheromone_grid.T, extent=(0, width, 0, height), 
                          origin='lower', cmap='YlGn', alpha=0.6, vmin=0, vmax=pheromone_intensity)

ax.legend()

# Initialize FSM states for agents
states = ["search"] * n_agents
best_targets = [None] * n_agents

def init():
    """Initialize animation."""
    agent_dots.set_data([], [])
    target_dots.set_data(targets[:, 0], targets[:, 1])
    obstacle_dots.set_data(obstacles[:, 0], obstacles[:, 1])
    pheromone_img.set_array(pheromone_grid.T)
    return agent_dots, target_dots, obstacle_dots, pheromone_img

def decay_pheromones():
    """Decay the pheromones over time."""
    global pheromone_grid
    pheromone_grid *= pheromone_decay

def update(frame):
    """Update the positions of agents and apply the FSM."""
    global agents, velocities, targets, pheromone_grid, states, best_targets
    
    decay_pheromones()

    for i in range(n_agents):
        # Finite State Machine for Agents
        if states[i] == "search":
            # Find nearest visible target
            distances = np.linalg.norm(targets - agents[i], axis=1)
            visible = distances < visibility_radius
            if np.any(visible):
                target_idx = np.argmin(distances)
                best_targets[i] = targets[target_idx]
                states[i] = "transport"
            else:
                # Random movement
                velocities[i] = (np.random.rand(2) * 2 - 1) * max_velocity
        
        elif states[i] == "transport":
            if best_targets[i] is not None:
                direction = best_targets[i] - agents[i]
                if np.linalg.norm(direction) < target_radius:
                    # Reached target
                    states[i] = "process"
                else:
                    velocities[i] = direction / np.linalg.norm(direction) * max_velocity
        
        elif states[i] == "process":
            # Simulate task processing
            if best_targets[i] is not None:
                pheromone_grid[int(best_targets[i][0]), int(best_targets[i][1])] += pheromone_intensity
                targets[np.all(targets == best_targets[i], axis=1)] = np.random.rand(2) * [width, height]
                states[i] = "search"
                best_targets[i] = None

        # Obstacle avoidance
        for obstacle in obstacles:
            if np.linalg.norm(agents[i] - obstacle) < agent_radius + 2:
                velocities[i] *= -1  # Simple bounce-back
        
        # Update agent positions
        agents[i] += velocities[i]
        agents[i] = np.clip(agents[i], 0, [width, height])
    
    # Update plot data
    agent_dots.set_data(agents[:, 0], agents[:, 1])
    target_dots.set_data(targets[:, 0], targets[:, 1])
    obstacle_dots.set_data(obstacles[:, 0], obstacles[:, 1])
    pheromone_img.set_array(pheromone_grid.T)

    return agent_dots, target_dots, obstacle_dots, pheromone_img

# Create animation
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)
plt.show()
