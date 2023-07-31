
# Q-Learning with Cartpole Problem of OpenAI Gym

### Author: Magi Chen <chenmagi@gmail.com>

This script implements the Q-Learning algorithm to solve the Cartpole problem from the OpenAI Gym environment.
The goal is to teach an agent to balance a pole on a moving cart for as long as possible.

## Environment:
- Action Space: The agent can take two actions - pushing the cart to the left or the right.
- Observation Space: The observations include the cart's position, velocity, pole angle, and pole angular velocity.

## Key Components:
1. MappingConfig: A class that helps to map and discretize the continuous observation values to discrete states.

2. LearningRateControl: A class that controls the exploration and learning rates during training.

3. Q_Learning: The main class that encapsulates the Q-Learning algorithm. It includes methods to select actions,
   update Q-values, and run the Q-Learning process.

## Usage:
1. Ensure you have the necessary dependencies installed: gym, numpy, matplotlib.
   You can install them using pip:

   ```
   pip install -r requirements.txt
   ```
