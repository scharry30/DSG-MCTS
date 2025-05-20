
import math
import random
from typing import List

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
    
    def add_child(self, child):
        self.children.append(child)
    
    def update(self, value):
        self.visits += 1
        self.value += (value - self.value) / self.visits

class MCTS:
    def __init__(self, mdp, max_iterations, uct_constant=1.0):
        self.mdp = mdp
        self.max_iterations = max_iterations
        self.uct_constant = uct_constant
    
    def uct_value(self, node):
        if node.visits == 0:
            return float('inf')
        return node.value + self.uct_constant * math.sqrt(math.log(node.parent.visits) / node.visits)
    
    def select(self, node):
        while node.children:
            node = max(node.children, key=self.uct_value)
        return node
    
    def expand(self, node):
        actions = self.mdp.get_possible_actions(node.state)
        for action in actions:
            next_state = self.mdp.get_next_state(node.state, action)
            child_node = Node(next_state, parent=node)
            node.add_child(child_node)
    
    def simulate(self, node):
        total_reward = 0
        current_state = node.state
        while True:
            action = random.choice(self.mdp.get_possible_actions(current_state))
            reward = self.mdp.get_reward(current_state, action)
            total_reward += reward
            current_state = self.mdp.get_next_state(current_state, action)
            if random.random() < 0.1:
                break
        return total_reward

    def backpropagate(self, node, reward):
        while node:
            node.update(reward)
            node = node.parent
    
    def run(self):
        root = Node(self.mdp.states[0])
        for _ in range(self.max_iterations):
            leaf = self.select(root)
            self.expand(leaf)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)
        
        return self.best_action(root)
    
    def best_action(self, node):
        best_child = max(node.children, key=lambda x: x.visits)
        return best_child.state
