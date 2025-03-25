import mesa
import math
import random
import numpy as np
import mesa.time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
print(sys.version)

class Bird(mesa.Agent):
    def __init__(self, unique_id, model, alpha, beta, gamma, delta, r0):
        super().__init__(unique_id, model)
        self.alpha = alpha  # Utility parameter for cohesion
        self.beta = beta    # Utility parameter for congestion
        self.gamma = gamma  # Utility parameter for alignment
        self.delta = delta  # Utility parameter for competition
        self.r0 = r0        # Neighborhood radius
        self.utility = 0    # Current utility
        self.neighbors = [] # List of neighbors
        self.num_neighbors = 0  # Number of neighbors
        self.alignment = 0  # Average alignment with neighbors
        self.position = (random.uniform(0, self.model.width), random.uniform(0, self.model.height))
        self.velocity = (random.uniform(-1, 1), random.uniform(-1, 1))  # Random initial velocity

    def find_neighbors(self):
        """Find neighbors within the neighborhood radius."""
        self.neighbors = [
            agent for agent in self.model.schedule.agents
            if agent != self and self.distance_to(agent) < self.r0
        ]
        self.num_neighbors = len(self.neighbors)

    def calculate_alignment(self):
        """Calculate the average alignment with neighbors."""
        if self.num_neighbors > 0:
            total_alignment = np.array([0.0, 0.0])
            for neighbor in self.neighbors:
                total_alignment += np.array(neighbor.velocity)
            self.alignment = total_alignment / self.num_neighbors
        else:
            self.alignment = np.array([0.0, 0.0])

    def calculate_utility(self):
        """Calculate the utility based on neighbors."""
        if self.num_neighbors > 0:
            cohesion_term = self.alpha * self.num_neighbors
            congestion_term = self.beta * (self.num_neighbors ** 2)
            alignment_term = self.gamma * self.num_neighbors * np.linalg.norm(self.alignment)
            competition_term = self.delta * math.log(self.num_neighbors)
            self.utility = cohesion_term - congestion_term + alignment_term - competition_term
        else:
            self.utility = 0  # If no neighbors, utility is 0

    def move(self):
        """Move the agent based on Reynolds rules and utility."""
        cohesion_vector = self.cohesion()
        separation_vector = self.separation()
        alignment_vector = self.alignment

        self.find_neighbors()

        #circular constraint:
        local_center = self.model.local_center_of_mass(self.neighbors)

        if local_center is not None:
            # scalar distance from agent to center
            distance_to_center = np.linalg.norm(local_center - np.array(self.position))
            # unit vector pointing from agent to center
            direction_to_center = (local_center - np.array(self.position)) / distance_to_center
            scaling_factor = ((distance_to_center - self.r0) / self.r0)**2
            local_flocking_force = direction_to_center * scaling_factor
        else:
            local_flocking_force = np.array([0.0, 0.0]) #no local flocking force if there are no neighbors

        # Combine Reynolds rules with utility-driven behavior
        utility_vector = (self.utility / 23.8) * (cohesion_vector + separation_vector + alignment_vector)

        total_force = utility_vector + 0.2 * local_flocking_force # changing the scaling for the global_flocking_force depending on how dominant we want it to be (seems like scaling factor >= 0.1 works besgt)

        # Update velocity and position
        # self.velocity = utility_vector / (np.linalg.norm(utility_vector)+1e-6) if np.linalg.norm(utility_vector) > 0 else self.velocity
        v = total_force / (np.linalg.norm(total_force)+1e-6) if np.linalg.norm(total_force) > 0 else self.velocity
        # print(v, len(self.neighbors))
        self.velocity = np.sign(v) * (np.abs(v) * len(self.neighbors)**(-0.2)) if len(self.neighbors) else v

        self.position = (
            (self.position[0] + self.velocity[0]) % self.model.width,
            (self.position[1] + self.velocity[1]) % self.model.height
        )

    def cohesion(self):
        """Cohesion rule: move towards the center of mass of neighbors."""
        if self.num_neighbors > 0:
            center_of_mass = np.mean([neighbor.position for neighbor in self.neighbors], axis=0)
            return (center_of_mass - np.array(self.position)) / np.linalg.norm(center_of_mass - np.array(self.position))
        return np.array([0.0, 0.0])

    def separation(self):
        """Separation rule: avoid crowding neighbors."""
        separation_force = np.array([0.0, 0.0])
        for neighbor in self.neighbors:
            distance = self.distance_to(neighbor)
            if distance < 2:  # If too close, move away
                separation_force -= (np.array(neighbor.position) - np.array(self.position)) / (distance ** 2 + 1e-6)
        return separation_force

    def distance_to(self, other):
        """Calculate the Euclidean distance to another agent."""
        return np.linalg.norm(np.array(self.position) - np.array(other.position))

    def step(self):
        """Execute one step of the agent's behavior."""
        self.find_neighbors()
        self.calculate_alignment()
        self.calculate_utility()
        self.move()


class FlockingModel(mesa.Model):
    def __init__(self, N, width, height, alpha, beta, gamma, delta, r0):
        super().__init__()
        self.num_agents = N
        self.width = width
        self.height = height
        self.schedule = mesa.time.RandomActivation(self)
        self.agents = []
        self.agent_positions = []

        # Create agents
        for i in range(self.num_agents):
            bird = Bird(i, self, alpha, beta, gamma, delta, r0)
            self.schedule.add(bird)
            self.agents.append(bird)

    def local_center_of_mass(self, neighbors):
        if neighbors:
            positions = np.array([neighbor.position for neighbor in neighbors])
            return np.mean(positions, axis= 0)
        else:
            return None #if no neighbors present

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        # Store positions after all agents have moved.
        current_positions = [(agent.position[0], agent.position[1]) for agent in self.agents]
        self.agent_positions.append(current_positions)


# # Visualization
# def update(frame, model: FlockingModel, scatter):
#     """Update function for Matplotlib animation."""
#     model.step()
#     x_vals = [agent.position[0] for agent in model.agents]
#     y_vals = [agent.position[1] for agent in model.agents]
#     scatter.set_offsets(np.c_[x_vals, y_vals])
#     return scatter,


# def run_visualization(model):
#     """Run Matplotlib animation."""
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_xlim(0, model.width)
#     ax.set_ylim(0, model.height)
#     scatter = ax.scatter([], [], color="blue", marker="o")

#     ani = animation.FuncAnimation(fig, update, frames=200, fargs=(model, scatter), interval=20)
#     plt.title("Flocking Simulation")
#     plt.show()

def run_and_plot(model: FlockingModel, frames=500, plot_frames=[1, 10, 50, 100, 150, 200, 500]):
    """Run the simulation and plot specified frames."""

    # Run the simulation and collect data
    for i in range(frames):
        print(f"Step {i}")
        model.step()

    # Create subplots
    fig, axes = plt.subplots(1, len(plot_frames), figsize=(5 * len(plot_frames), 5))
    if len(plot_frames) == 1:  # Avoid indexing issues with a single subplot
        axes = [axes]

    # Plot the specified frames
    for i, frame_num in enumerate(plot_frames):
        print(frame_num)
        if frame_num > frames or frame_num < 1:
            print(f"Warning: Frame {frame_num} is out of range and will be skipped.")
            continue
        
        ax = axes[i]
        positions = model.agent_positions[frame_num - 1]
        x_vals, y_vals = zip(*positions)

        ax.scatter(x_vals, y_vals, color="blue", marker="o")
        ax.set_xlim(0, model.width)
        ax.set_ylim(0, model.height)
        ax.set_title(f"Frame {frame_num}")

    plt.tight_layout()
    plt.show()


# Parameters
N = 200
width, height = 100, 100
alpha = 0.5
beta = 0.005
gamma = 0.25
delta = 1
r0 = 3

# Create and run the model
model = FlockingModel(N, width, height, alpha, beta, gamma, delta, r0)
# run_visualization(model)
run_and_plot(model, frames=500, plot_frames=[1, 10, 50, 100, 500])