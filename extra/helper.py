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
            pos = np.array([i, j])
            # check if the position is unoccupied and adjacent to the tree, we also need to check if the position is not the agent's position 
            if not any(np.array_equal(pos, occupied) for occupied in occupied_positions) and not np.array_equal(pos, agent_pos) and is_adjacent(pos, tree_position):
                if closest_tile is None:
                    closest_tile = [i, j]
                elif manhattan_distance([i, j], agent_pos) < manhattan_distance(closest_tile, agent_pos):
                    closest_tile = [i, j]
    return closest_tile

def testClosestUnoccupiedTile():
    tree_position = [3, 3]
    agent_pos = [5, 0]
    occupied_positions = [[4, 2], [3,2], [4,3], [2,2]]
    size = 6
    print(closestUnoccupiedTile(tree_position, agent_pos, occupied_positions, size))

def test_is_adjacent():
    print(is_adjacent([3, 3], [4, 2]))
    print(is_adjacent([3, 3], [3, 2]))
    print(is_adjacent([3, 3], [2, 2]))
    print(is_adjacent([3, 3], [4, 4]))
    print(is_adjacent([3, 3], [3, 4]))
    print(is_adjacent([3, 3], [2, 4]))
    print(is_adjacent([3, 3], [2, 3]))
    print(is_adjacent([3, 3], [4, 3]))

#test_is_adjacent()
#testClosestUnoccupiedTile()