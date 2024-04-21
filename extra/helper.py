import numpy as np

# returns the distance between two objects in a grid excluding diagonals
def manhattan_distance(obj1, obj2):
    return abs(obj1[0] - obj2[0]) + abs(obj1[1] - obj2[1])

# returns the distance between two objects in a grid including diagonals
def chebyshev_distance(obj1, obj2):
    return max(abs(obj1[0] - obj2[0]), abs(obj1[1] - obj2[1]))

def is_adjacent(obj1, obj2):
    return chebyshev_distance(obj1, obj2) == 1

# implement closest tile adjacent to tree
def closestUnoccupiedTile(tree_position, agent_pos, occupied_positions, size):
    closest_tile = None
    for i in range(size):
        for j in range(size):
            if [i, j] not in occupied_positions and is_adjacent([i, j], tree_position):
                if closest_tile is None:
                    closest_tile = [i, j]
                elif manhattan_distance([i, j], agent_pos) < manhattan_distance(closest_tile, agent_pos):
                    closest_tile = [i, j]
    return closest_tile
