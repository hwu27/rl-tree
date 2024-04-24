# rl_tree

Implementation of a Reinforcement Learning Challenge using:

- gymnasium (previously Openai Gym)
- DQN with Tensorflow Agents
- Numpy 

Environment:

1. The environment consists of a tileset of size - 1 * size - 1 (index starts at 0)
2. There is a tree at [(size/2), (size/2)] 
3. The agent is at [(size - 1), (0)] 

Note, the tiles are in the format (y, x), where y goes down the rows and x goes right down the columns

4. The goal is to train a model that points the agent to land on the adjacent tiles

Rules:
1. When an agent lands on an adjacent tile, that tile is called "occupied" and the agent will go back to start
2. When an agent lands on a previously landed "occupied" tile, the agent will be reset
3. When the agent lands on 4 adjacent tiles, you win (can be increased)
4. The agent cannot move diagonally
5. The agent cannot move with modulo based actions (going to the left at the edge and then wrapping around to the right)

If you want to attempt the challenge, use the empty_parking.py
