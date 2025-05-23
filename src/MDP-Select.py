import random
import math
import numpy as np
from collections import defaultdict


class MDPStrategySelector:
    def __init__(self, strategies, transition_probabilities, rewards, gamma=0.9, epsilon=1e-6):
        self.strategies = strategies
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_function = defaultdict(lambda: 0)
        self.policy = {}

    def compute_value_function(self, max_iterations=1000):
        iteration = 0
        while iteration < max_iterations:
            delta = 0
            for state in self.strategies:
                max_value = float('-inf')
                for action in self.strategies:
                    expected_value = 0
                    for next_state in self.strategies:
                        transition_prob = self.transition_probabilities.get((state, action, next_state), 0)
                        reward = self.rewards.get((state, action, next_state), 0)
                        expected_value += transition_prob * (reward + self.gamma * self.value_function[next_state])
                    max_value = max(max_value, expected_value)
                delta = max(delta, abs(self.value_function[state] - max_value))
                self.value_function[state] = max_value
            if delta < self.epsilon:
                break
            iteration += 1

    def compute_optimal_policy(self):
        for state in self.strategies:
            best_action = None
            max_value = float('-inf')
            for action in self.strategies:
                expected_value = 0
                for next_state in self.strategies:
                    transition_prob = self.transition_probabilities.get((state, action, next_state), 0)
                    reward = self.rewards.get((state, action, next_state), 0)
                    expected_value += transition_prob * (reward + self.gamma * self.value_function[next_state])
                if expected_value > max_value:
                    max_value = expected_value
                    best_action = action
            self.policy[state] = best_action

    def select_best_strategy(self, current_state):
        return self.policy.get(current_state, None)

    def display_policy(self):
        for state, action in self.policy.items():
            print(f"State: {state}, Best Strategy: {action}")


class MCTS_MDP_Integration:
    def __init__(self, mcts_task, mdp_strategy_selector):
        self.mcts_task = mcts_task
        self.mdp_strategy_selector = mdp_strategy_selector

    def integrate_mdp(self):
        root = treeNode('')
        max_iterations = self.mcts_task.iteration_limit
        for i in range(max_iterations):
            print(f'Iteration {i+1} starting...\n')
            flag, node, root = executeRound(root, self.mcts_task)
            if flag:
                print('Solution found.\n')
                return root, node
        print('Maximum iterations reached.\n')
        return root, None

    def executeRound(self, root, mcts_task):
        print('-' * 40)
        print('Selection Phase\n')
        flag, node = selectNode(root, mcts_task)
        if flag:
            if mcts_task.sample_value != 'full':
                return True, node, root
            else:
                node.reflection = '<end>'
        
        print('-' * 40)
        print('Expansion Phase\n')
        if node.reflection == '<end>':
            print('Skipping this phase.\n')
        else:
            node = self.expand_node(node, mcts_task)

        print('-' * 40)
        print('Simulation Phase\n')
        if node.reflection == '<end>':
            print('Skipping simulation.\n')
        else:
            best_strategy = self.mdp_strategy_selector.select_best_strategy(node.y)
            if best_strategy:
                roll_node = getBestChild(node, mcts_task)
                best_V = self.execute_policy(roll_node, best_strategy, mcts_task)
                roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
                roll_node.numVisits += 1
        
        print('-' * 40)
        print('Backpropagation Phase\n')
        self.back_propagate(node)
        return False, node, root

    def execute_policy(self, node, strategy, mcts_task):
        max_V = mcts_task.low
        strs = node.y
        cur_step = node.depth + 1
        if mcts_task.use_reflection == 'common':
            reflection = mcts_task.get_reflection(strs, cur_step)
        else:
            reflection = mcts_task.get_simple_reflection(strs, cur_step)
        node.update_reflection(reflection)
        if reflection == '<end>':
            print('This step has been resolved and does not require simulation.\n')
            return node.V
        for i in range(mcts_task.roll_forward_steps):
            next_steps = get_next_steps_roll(strs, cur_step, mcts_task)
            if not next_steps:
                break
            action = random.choice(next_steps)
            strs = strs + action
            cur_step += 1
            value = mcts_task.get_step_value(strs)
            if value > max_V:
                max_V = value
            if mcts_task.use_reflection == 'common':
                cur_ref = mcts_task.get_reflection(strs, cur_step)
            else:
                cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
            if cur_ref == '<end>':
                break
        return max_V

    def expand_node(self, node, mcts_task):
        if not node.reflection:
            reflection = mcts_task.get_reflection(node.y, node.depth + 1)
            node.update_reflection(reflection)
        if node.reflection == '<end>':
            return node
        actions = get_next_steps_expand(node, mcts_task)
        if not actions:
            node.update_reflection('<end>')
            return node
        for action in actions:
            if action not in node.children.keys():
                node.append_children(action)
                child = node.children[action]
                value = mcts_task.get_step_value(child.y)
                child.update_value(value)
                if mcts_task.sample_value == 'full':
                    child.update_reflection(mcts_task.get_reflection(child.y, child.depth + 1))
                child.visit_sequence = mcts_task.node_count
                mcts_task.update_count()
        node.isFullyExpanded = True
        return node

    def back_propagate(self, node):
        while node is not None:
            node.numVisits += 1
            if node.isFullyExpanded:
                child_Vs = [child.V * child.numVisits for child in node.children.values()]
                total_num_visits = sum([child.numVisits for child in node.children.values()])
                if total_num_visits > 0:
                    node.V = sum(child_Vs) / total_num_visits
            node = node.parent


class treeNode:
    def __init__(self, y):
        self.y = y
        self.V = 0
        self.numVisits = 0
        self.children = {}
        self.parent = None
        self.depth = 0
        self.reflection = None
        self.isFullyExpanded = False
        self.visit_sequence = 0

    def update_reflection(self, reflection):
        self.reflection = reflection

    def update_value(self, value):
        self.V = value

    def append_children(self, action):
        self.children[action] = treeNode(self.y + action)
        self.children[action].parent = self
        self.children[action].depth = self.depth + 1


def selectNode(node, mcts_task):
    while node.isFullyExpanded:
        node = getBestChild(node, mcts_task)
    if mcts_task.reward_model_type == 'vm':
        return node.final_ans_flag == 1, node
    else:
        return False, node


def getBestChild(node, mcts_task):
    bestValue = mcts_task.low
    bestNodes = []
    for child in node.children.values():
        nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
            2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)


def get_next_steps_roll(y, step_n, mcts_task):
    next_steps = []
    for i in range(mcts_task.roll_branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_step(y, step_n)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps


def get_next_steps_expand(node, mcts_task):
    next_steps = []
    for i in range(mcts_task.branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_step(node.y, node.depth + 1)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps
