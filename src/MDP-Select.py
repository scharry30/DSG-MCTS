import random
import math
import numpy as np
from collections import defaultdict
from utils.node_utils import Node, backpropagate, select_leaf


class MDPStrategySelector:


    def __init__(self, strategies, gamma=0.9, epsilon=1e-6):

        self.strategies = strategies
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_function = defaultdict(lambda: 0)
        self.policy = {}


        self.transition_probabilities = {}
        self.rewards = {}


        self.strategy_counts = {s.name: 0 for s in strategies}
        self.strategy_success = {s.name: 0 for s in strategies}

    def update_transition_reward(self, current_state, action, next_state, reward):

        key = (current_state, action.name, next_state)
        if key not in self.transition_probabilities:
            self.transition_probabilities[key] = 0
        self.transition_probabilities[key] += 1

        if key not in self.rewards:
            self.rewards[key] = reward
        else:
            self.rewards[key] = 0.9 * self.rewards[key] + 0.1 * reward


        total = sum(self.transition_probabilities.get((current_state, action.name, s), 0)
                    for s in self.get_possible_states())
        for s in self.get_possible_states():
            norm_key = (current_state, action.name, s)
            if norm_key in self.transition_probabilities:
                self.transition_probabilities[norm_key] /= total

    def get_possible_states(self):

        return [s.name for s in self.strategies]

    def compute_value_function(self, max_iterations=100):

        states = self.get_possible_states()
        iteration = 0
        while iteration < max_iterations:
            delta = 0
            for state in states:
                old_value = self.value_function[state]
                max_value = float('-inf')

                for strategy in self.strategies:
                    expected_value = 0
                    for next_state in states:
                        transition_prob = self.transition_probabilities.get((state, strategy.name, next_state), 0)
                        if transition_prob > 0:  # 只考虑有转移概率的状态
                            reward = self.rewards.get((state, strategy.name, next_state), 0)
                            expected_value += transition_prob * (reward + self.gamma * self.value_function[next_state])

                    if expected_value > max_value:
                        max_value = expected_value


                if max_value == float('-inf'):
                    max_value = 0

                self.value_function[state] = max_value
                delta = max(delta, abs(old_value - max_value))

            if delta < self.epsilon:
                break
            iteration += 1

    def compute_optimal_policy(self):

        states = self.get_possible_states()
        for state in states:
            best_action = None
            max_value = float('-inf')

            for strategy in self.strategies:
                expected_value = 0
                for next_state in states:
                    transition_prob = self.transition_probabilities.get((state, strategy.name, next_state), 0)
                    if transition_prob > 0:
                        reward = self.rewards.get((state, strategy.name, next_state), 0)
                        expected_value += transition_prob * (reward + self.gamma * self.value_function[next_state])

                if expected_value > max_value:
                    max_value = expected_value
                    best_action = strategy


            if best_action is None:
                best_action = self.get_best_strategy_by_success_rate(state)

            self.policy[state] = best_action

    def get_best_strategy_by_success_rate(self, state):

        success_rates = {}
        for strategy in self.strategies:
            count = self.strategy_counts.get(strategy.name, 0)
            if count > 0:
                success = self.strategy_success.get(strategy.name, 0)
                success_rates[strategy] = success / count
            else:
                success_rates[strategy] = 0


        for strategy in self.strategies:
            count = self.strategy_counts.get(strategy.name, 0)
            exploration_bonus = 1.0 / (count + 1)
            success_rates[strategy] += exploration_bonus

        return max(success_rates.items(), key=lambda x: x[1])[0]

    def select_strategy(self, state, exploration_rate=0.1):

        if not self.policy or random.random() < exploration_rate:
            return random.choice(self.strategies)


        if state not in self.policy:
            return self.get_best_strategy_by_success_rate(state)

        return self.policy[state]

    def update_strategy_statistics(self, strategy_name, success):

        if strategy_name in self.strategy_counts:
            self.strategy_counts[strategy_name] += 1
            if success:
                self.strategy_success[strategy_name] += 1

    def get_strategy_success_rates(self):

        success_rates = {}
        for strategy_name in self.strategy_counts:
            count = self.strategy_counts[strategy_name]
            if count > 0:
                success = self.strategy_success[strategy_name]
                success_rates[strategy_name] = success / count
            else:
                success_rates[strategy_name] = 0
        return success_rates


class DSGMCTSWithMDP:

    def __init__(self, task, strategies, exploration_weight=1.0):

        self.task = task
        self.strategies = strategies
        self.exploration_weight = exploration_weight
        self.mdp_selector = MDPStrategySelector(strategies)


        self.root = Node(state=task.initial_state)

    def search(self, num_iterations=100):

        for i in range(num_iterations):

            leaf = select_leaf(self.root, self.exploration_weight)

            if not self.task.is_terminal(leaf.state):
                self._expand(leaf)

            if leaf.children:
                child = self._select_child_with_mdp(leaf)
                value = self._simulate(child)


                backpropagate(child, value)


            if i > 0 and i % 10 == 0:
                self._update_mdp_model()


        best_child = self._select_best_child(self.root)
        return self.root, best_child

    def _expand(self, node):

        state = node.state


        problem_state = self._get_problem_state(state)


        strategy = self.mdp_selector.select_strategy(problem_state)


        actions = self.task.get_actions(state, strategy)


        for action in actions:
            next_state = self.task.get_next_state(state, action)
            child = Node(
                state=next_state,
                parent=node,
                value=0.0,
                strategy=strategy,
                action=action
            )
            node.add_child(child)

    def _select_child_with_mdp(self, node):

        problem_state = self._get_problem_state(node.state)


        strategy = self.mdp_selector.select_strategy(problem_state)

        strategy_children = [child for child in node.children if child.strategy == strategy]

        if strategy_children:

            return max(strategy_children, key=lambda c: c.value / max(c.visits, 1) +
                                                        self.exploration_weight * math.sqrt(
                math.log(node.visits + 1) / max(c.visits, 1)))
        else:

            return max(node.children, key=lambda c: c.value / max(c.visits, 1) +
                                                    self.exploration_weight * math.sqrt(
                math.log(node.visits + 1) / max(c.visits, 1)))

    def _simulate(self, node):
        """执行模拟"""
        state = node.state
        depth = 0
        max_depth = 8 

        current_strategy = node.strategy

        while not self.task.is_terminal(state) and depth < max_depth:

            problem_state = self._get_problem_state(state)


            if random.random() < 0.3:
                current_strategy = self.mdp_selector.select_strategy(problem_state)


            actions = self.task.get_actions(state, current_strategy)
            if not actions:
                break


            action = random.choice(actions)


            state = self.task.get_next_state(state, action)
            depth += 1


        value = self.task.evaluate(state)


        success = value > 0.7  
        self.mdp_selector.update_strategy_statistics(current_strategy.name, success)

        return value

    def _update_mdp_model(self):

        self._collect_transition_data(self.root)


        self.mdp_selector.compute_value_function()
        self.mdp_selector.compute_optimal_policy()

    def _collect_transition_data(self, node):

        if not node.children:
            return

        current_state = self._get_problem_state(node.state)

        for child in node.children:
            if child.strategy:

                next_state = self._get_problem_state(child.state)


                reward = child.value

                self.mdp_selector.update_transition_reward(
                    current_state,
                    child.strategy,
                    next_state,
                    reward
                )


            self._collect_transition_data(child)

    def _get_problem_state(self, state):

        if isinstance(state, str):
            return state[:20]
        return str(state)[:20]

    def _select_best_child(self, node):

        if not node.children:
            return None


        return max(node.children, key=lambda c: c.visits)


def run_dsg_mcts_with_mdp(task, strategies, num_iterations=100, exploration_weight=1.0):

    mcts = DSGMCTSWithMDP(task, strategies, exploration_weight)
    return mcts.search(num_iterations)
