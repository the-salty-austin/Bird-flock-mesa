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

        # Combine Reynolds rules with utility-driven behavior
        utility_vector = (self.utility / 23.8) * (cohesion_vector + separation_vector + alignment_vector)

        # Update velocity and position
        self.velocity = utility_vector / (np.linalg.norm(utility_vector)+1e-6) if np.linalg.norm(utility_vector) > 0 else self.velocity
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

        # Create agents
        for i in range(self.num_agents):
            bird = Bird(i, self, alpha, beta, gamma, delta, r0)
            self.schedule.add(bird)
            self.agents.append(bird)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()


# Visualization
def update(frame, model, scatter):
    """Update function for Matplotlib animation."""
    model.step()
    x_vals = [agent.position[0] for agent in model.agents]
    y_vals = [agent.position[1] for agent in model.agents]
    scatter.set_offsets(np.c_[x_vals, y_vals])
    return scatter,


def run_visualization(model):
    """Run Matplotlib animation."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)
    scatter = ax.scatter([], [], color="blue", marker="o")

    ani = animation.FuncAnimation(fig, update, frames=200, fargs=(model, scatter), interval=20)
    plt.title("Flocking Simulation")
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
run_visualization(model)