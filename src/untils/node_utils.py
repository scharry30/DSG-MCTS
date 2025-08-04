import math
import random


class Node:
    def __init__(self, state=None, parent=None, value=0.0, strategy=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = value
        self.strategy = strategy
        self.action = action

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    def update(self, value):
        self.visits += 1
        self.value = ((self.value * (self.visits - 1)) + value) / self.visits

    def is_leaf(self):
        return len(self.children) == 0

    def is_fully_expanded(self, branching_factor):
        return len(self.children) >= branching_factor

    def best_child(self, exploration_weight=1.0):
        if not self.children:
            return None

        choices_weights = [
            (child.value / child.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits)) / child.visits)
            if child.visits > 0 else float('inf')
            for child in self.children
        ]

        return self.children[choices_weights.index(max(choices_weights))]

    def random_child(self):
        if not self.children:
            return None
        return random.choice(self.children)

    def get_path(self):
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]

    def get_depth(self):
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth


def backpropagate(node, value):
    while node is not None:
        node.update(value)
        node = node.parent


def select_leaf(root, exploration_weight=1.0):
    node = root
    while not node.is_leaf():
        if not all(child.visits > 0 for child in node.children):
            unvisited = [child for child in node.children if child.visits == 0]
            return random.choice(unvisited)
        else:
            node = node.best_child(exploration_weight)
    return node
