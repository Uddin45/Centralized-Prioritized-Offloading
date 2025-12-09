Prioritized Deep Q-Network for Vehicular Edge Computing

Vehicular Edge Computing (VEC) enables real-time decision-making by offloading computationally intensive vehicular tasks to roadside edge servers. This project investigates task offloading and scheduling optimization in VEC, with a strong focus on task prioritization to improve system responsiveness and reliability.

We develop a Prioritized Deep Q-Network (DQNP) that learns optimal offloading decisions by maximizing long-term cumulative rewards under dynamic vehicular environments. The model integrates a priority-scaled reward system that assigns differentiated importance to tasks based on their priority level, enabling the agent to:

Maximize task completion within deadlines

Minimize latency and energy consumption

Ensure balanced offloading performance across all priority classes

The DQNP intelligently adapts task-selection behavior to changing environmental conditions. For example, in poor channel states, the agent prioritizes tasks with higher deadlines to maintain fairness and stability across the system. This dynamic adjustment promotes efficient offloading decisions for heterogeneous vehicular workloads.

This repository includes:

Simulation environment for multi-priority vehicular task offloading

Implementation of the prioritized DQN architecture

Reward formulation incorporating latency, energy, and deadlines

Training scripts and performance evaluation tools

Our results demonstrate that prioritized reinforcement learning can significantly enhance VEC efficiency, scalability, and fairness across priority levels.
