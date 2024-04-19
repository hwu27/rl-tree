import numpy as np

# returns the distance between two objects in a grid
# only allows horizontal or vertical movement
def manhattan_distance(obj1, obj2):
    return abs(obj1[0] - obj2[0]) + abs(obj1[1] - obj2[1])

def is_adjacent(obj1, obj2):
    return manhattan_distance(obj1, obj2) == 1

# implement closest tile adjacent to tree

