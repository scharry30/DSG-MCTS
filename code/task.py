import random
import time
import math
from tasks.science import SearchTask
from MCTS.base import treeNode
from models.get_response import get_proposal
from MCTS.mcts import MCTS
from utils.verify_MATH import exact_match_score, grade_answer, extract_answer
from utils.verify_llm import llm_verify
from utils.solution_summary_extractor import extract_summary_from_solution


class MCTS_Task(SearchTask):
    def __init__(self, data, propose_method='llama', value_method='glm', branch=3, end_gate=0.9, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=3, time_limit=None, iteration_limit=None, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, use_reflection='simple', low=0, high=1,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', lang='zh', weighted_verify=False):
        super().__init__(data, propose_method, value_method)
        
        # Task initialization
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.branch = branch
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.end_gate = end_gate
        self.use_reflection = use_reflection
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1
        self.sample_value = sample_value
        self.answer = answer
        self.verify_method = verify_method
        self.reward_model_type = 'prm' if USE_PRM else 'vm'
        self.lang = lang
        self.weighted_verify = weighted_verify

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def set_limit_type(self):
        if self.time_limit is not None:
            if self.iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            self.limit_type = 'time'
        else:
            if self.iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = 'iterations'

    def get_next_step(self, y, step_n):
        prompt = self.get_propose_prompt(y, step_n)
        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length, self.truncation, self.do_sample, self.max_new_tokens)
        return self.process_response(response, y, step_n)

    def get_propose_prompt(self, y, step_n):
        if self.use_case_prompt:
            return self.single_propose_prompt_wrap(self.question, y, step_n)
        if self.propose_method in ['gpt', 'llama']:
            return self.zero_single_propose_wrap_mistral(self.question, y, step_n)
        return self.zero_single_propose_wrap(self.question, y, step_n, self.lang)

    def process_response(self, response, y, step_n):
        if not response:
            print('Error: No next step generated.\n')
            return ''
        
        p = ' '.join(response).strip()

        # Ensure response is unique and has adequate length
        if self.lang == 'zh':
            return self.process_chinese_response(p, y, step_n)
        return self.process_english_response(p, y, step_n)

    def process_chinese_response(self, response, y, step_n):
        if 'next:' in response:
            stp = response.split('next:')[1].strip()
            return self.validate_step(stp, y, step_n)

        if '步骤' in response and ':' in response:
            pre_len = len(response.split(':')[0])
            p_ = response[pre_len:].strip()
            return self.validate_step(p_, y, step_n)

        print('wrong!\n')
        return ''

    def process_english_response(self, response, y, step_n):
        if "Next step:" in response:
            stp = response.split('Next step:')[1].strip()
            return self.validate_step(stp, y, step_n)

        if "Step" in response and ":" in response:
            pre_len = len(response.split(':')[0])
            p_ = response[pre_len:].strip()
            return self.validate_step(p_, y, step_n)

        print('Error in output format!\n')
        return ''

    def validate_step(self, step, y, step_n):
        if len(step) < 2:
            print('Step too short!\n')
            return ''
        if step in y:
            print('Step repeated!\n')
            return ''
        return f'Step {step_n}: {step}\n'

    def get_next_step_use_reflection(self, y, step_n, reflection):
        if self.propose_method == 'gpt' or self.propose_method == 'local':
            propose_prompt = self.zero_single_propose_wrap_use_reflection_gpt(self.question, y, step_n, reflection)
        else:
            propose_prompt = self.zero_single_propose_wrap_use_reflection(self.question, y, step_n, reflection)
        response = get_proposal(propose_prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length, self.truncation, self.do_sample, self.max_new_tokens)
        return self.process_response(response, y, step_n)

    def get_step_value(self, y):
        if y in self.value_cache:
            return self.value_cache[y]

        value = self.calculate_step_value(y)
        self.value_cache[y] = value
        return value

    def calculate_step_value(self, y):
        prompt_answer = self.generate_step_value_prompt(y)
        response = get_value(prompt_answer, self.value_method, self.temperature, self.max_tokens, self.seed,
                             self.max_length, self.low, self.high)
        return self.extract_value(response)

    def generate_step_value_prompt(self, y):
        if self.lang == 'zh':
            return f'q: {self.question}\nsetp: 【answer】{y}'
        return f'Problem: {self.question}\nSolution: {y}'

    def extract_value(self, response):
        return self.value_outputs_unwrap(response, self.low, self.high)

    def get_summary(self, y):
        if self.lang == 'zh':
            return self.process_chinese_summary(y)
        return self.process_english_summary(y)

    def process_chinese_summary(self, y):
        if self.evaluate == 'scibench':
            return self.evaluate_summary_prompt_wrap(self.question, y)
        return self.summary_prompt_wrap(self.question, y)

    def process_english_summary(self, y):
        prompt = self.MATH_summary_prompt_wrap(self.question, y)
        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length, self.truncation, self.do_sample, 128)
        return self.extract_summary(response)

    def extract_summary(self, response):
        summary = ' '.join(response).strip()
        print(f'Generated summary: {summary}\n')
        return summary

    def get_final_solution(self, root, weighted):
        end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate) if self.reward_model_type == 'vm' else root.get_all_end_root_nodes_prm()
        return self.select_best_solution(end_leaf_nodes, weighted)

    def select_best_solution(self, end_leaf_nodes, weighted):
        if not end_leaf_nodes or not weighted:
            best_node, best_V = root.getBestV()
            solution = best_node.y
            return solution, self.get_summary(solution)

        all_answers = {}
        for leaf in end_leaf_nodes:
            solution = leaf.y
            summ = leaf.summary
            extracted_answer = extract_answer(summ)
            all_answers[extracted_answer] = [solution, summ, leaf.V]

        best_answer = max(all_answers.values(), key=lambda x: x[2])
        return best_answer[0], best_answer[1]

    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, finish, root = MCTS(self)
        solution, summ = self.get_final_solution(root, self.weighted_verify)
        result = exact_match_score(summ, self.answer)
        return {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish, 'accurate': result, 'real_answer': self.answer}
