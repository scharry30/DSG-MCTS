import random
import numpy
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference_and_value_models import get_inference_model_llama, local_inference_model

# Function to simulate the MCTS search using Llama model
def MCTS_search(mcts_task, inference_model, inference_tokenizer):
    root = treeNode('')
    if mcts_task.limit_type == 'time':
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        while time.time() < timeLimit:
            flag, node, root = executeRound(root, mcts_task, inference_model, inference_tokenizer)
            if flag:
                return root, node, time.time() - time_start
    else:
        for i in range(mcts_task.iteration_limit):
            flag, node, root = executeRound(root, mcts_task, inference_model, inference_tokenizer)
            if flag:
                return root, node, i + 1
    return root, None, None

# Function to execute a round of the MCTS
def executeRound(root, mcts_task, inference_model, inference_tokenizer):
    flag, node = selectNode(root, mcts_task)
    if flag:
        if mcts_task.sample_value != 'full':
            return True, node, root
        else:
            node.reflection = '<end>'

    node = expand(node, mcts_task, inference_model, inference_tokenizer)

    if mcts_task.reward_model_type == 'vm':
        roll_node = getBestChild(node, mcts_task)
        best_V = local_inference_model(roll_node, inference_model, inference_tokenizer)
        roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
        roll_node.numVisits += 1

    back_propagate(node)
    return False, node, root

# Function for the expansion phase of the tree
def expand(node, mcts_task, inference_model, inference_tokenizer):
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
            child.visit_sequence = mcts_task.node_count
            mcts_task.update_count()
    node.isFullyExpanded = True
    return node

# Backpropagate the result through the tree
def back_propagate(node):
    while node is not None:
        node.numVisits += 1
        if node.isFullyExpanded:
            child_Vs = [child.V * child.numVisits for child in node.children.values()]
            total_num_visits = sum([child.numVisits for child in node.children.values()])
            if total_num_visits > 0:
                node.V = sum(child_Vs) / total_num_visits
        node = node.parent

# Select the best child node during the selection phase
def selectNode(node, mcts_task):
    while node.isFullyExpanded:
        node = getBestChild(node, mcts_task)
    if isTerminal(node, mcts_task):
        node.final_ans_flag = 1
        return True, node
    else:
        return False, node

# Check if a node is terminal
def isTerminal(node, mcts_task):
    if mcts_task.reward_model_type == 'vm':
        return node.V >= mcts_task.end_gate
    else:
        return False
