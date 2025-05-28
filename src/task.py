import random
from tasks.science import SearchTask
from MCTS.base import treeNode
from models.get_response import *
from MCTS.mcts import MCTS
from utils.verify_MATH import exact_match_score, grade_answer, extract_answer
from utils.verify_llm import llm_verify
from utils.solution_summary_extractor import extract_summary_from_solution


class Strategy:
    def __init__(self, name, keywords=None, description=None):
        self.name = name
        self.keywords = keywords or []
        self.description = description or ""


class MCTS_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', branch=3, end_gate=0.9, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=3, time_limit=None, iteration_limit=None, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.9, max_tokens=8192, seed=170, max_length=8192, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, use_reflection='simple', low=0, high=1,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', lang='zh',
                 weighted_verify=False):
        super().__init__(data, propose_method, value_method)
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
        self.reward_model_type = 'vm'  # 只使用value model
        self.lang = lang
        self.weighted_verify = weighted_verify

        # DSG-MCTS 特定属性
        self.active_strategy = None
        self.strategy_weight = 0.5
        self.diversity_weight = 0.2
        self.high_value_paths = set()
        self.strategies = []
        self._initialize_strategies()

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
        if self.use_case_prompt:
            prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
        else:
            if self.propose_method == 'gpt':
                prompt = self.zero_single_propose_wrap_gpt(self.question, y, step_n, self.lang)
            elif self.propose_method == 'mistral' or self.propose_method == 'llama':
                prompt = self.zero_single_propose_wrap_mistral(self.question, y, step_n)
            else:
                prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)


        if self.active_strategy:
            strategy_info = f"\nUse {self.active_strategy.name} strategy: {self.active_strategy.description}"
            prompt += strategy_info

        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            return ''

        if len(response) > 5:
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '下一步:' in p:
                stp = p.split('下一步:')[1].strip()
                if len(stp) < 2:
                    return ''
                if stp in y:
                    return ''

                revised_ = '步骤' + str(step_n) + ':' + stp
                return revised_ + '\n'

            elif '步骤' in p and ':' in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('步骤')[0].strip()
                if len(p_) < 3:
                    return ''
                if p_[1:] in y:
                    return ''

                revised_ = '步骤' + str(step_n) + p_
                return revised_ + '\n'

            else:
                return ''

        else:
            if "Next step:" in p:
                stp = p.split('Next step:')[1].strip()
                if len(stp) < 2:
                    return ''
                if stp in y:
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + stp
                return revised_ + '\n'

            elif "Step" in p and ":" in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('Step')[0].strip()
                if len(p_) < 4:
                    return ''
                p_ = p_[1:].strip()
                if p_ in y:
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                return revised_ + '\n'

            else:
                p_ = p.strip()
                if len(p_) < 3:
                    return ''
                if p_ in y:
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                return revised_ + '\n'

    def get_next_step_use_reflection(self, y, step_n, reflection):
        if self.propose_method == 'gpt' or self.propose_method == 'local':
            propose_prompt = self.zero_single_propose_wrap_use_reflection_gpt(self.question, y, step_n, reflection,
                                                                              self.lang)
        else:
            propose_prompt = self.zero_single_propose_wrap_use_reflection(self.question, y, step_n, reflection,
                                                                          self.lang)


        if self.active_strategy:
            strategy_info = f"\nUse {self.active_strategy.name} strategy: {self.active_strategy.description}"
            propose_prompt += strategy_info

        response = get_proposal(propose_prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            return ''

        if len(response) > 5:
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '下一步:' in p:
                stp = p.split('下一步:')[1].strip()
                if len(stp) < 2:
                    return ''
                if stp in y:
                    return ''

                revised_ = '步骤' + str(step_n) + ':' + stp
                return revised_ + '\n'

            elif '步骤' in p and ':' in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('步骤')[0].strip()
                if len(p_) < 3:
                    return ''
                if p_[1:] in y:
                    return ''

                revised_ = '步骤' + str(step_n) + p_
                return revised_ + '\n'

            else:
                return ''

        else:
            if "Next step:" in p:
                stp = p.split('Next step:')[1].strip()
                if len(stp) < 2:
                    return ''
                if stp in y:
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + stp
                return revised_ + '\n'

            elif "Step" in p and ":" in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('Step')[0].strip()
                if len(p_) < 4:
                    return ''
                p_ = p_[1:].strip()
                if p_ in y:
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                return revised_ + '\n'

            else:
                return ''

    def get_simple_reflection(self, y, step_n):
        if step_n == 1:
            return '<continue>'
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer is' in y or '\\boxed' in y:
                return '<end>'

        if self.propose_method == 'mistral':
            reflection_prompt = self.single_reflection_wrap_simple_mistral(self.question, y, step_n)
        else:
            reflection_prompt = self.single_reflection_wrap_simple(self.question, y, step_n, self.lang)
        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            cnt -= 1
        if not response:
            return '<end>'

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '继续' in p:
                return '<continue>'
            elif '结束' in p:
                return '<end>'
            else:
                return '<continue>'
        else:
            if 'continue' in p.lower():
                return '<continue>'
            elif 'end' in p.lower() or 'done' in p.lower() or 'finish' in p.lower():
                return '<end>'
            else:
                return '<continue>'

    def get_reflection(self, y, step_n):
        if step_n == 1:
            return '<continue>'
        if self.propose_method in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer is' in y or '\\boxed' in y:
                return '<end>'

        if self.propose_method == 'gpt':
            reflection_prompt = self.single_reflection_wrap_gpt(self.question, y, step_n, self.lang)
        elif self.propose_method == 'mistral':
            reflection_prompt = self.single_reflection_wrap_mistral(self.question, y, step_n)
        else:
            reflection_prompt = self.single_reflection_wrap(self.question, y, step_n, self.lang)
        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            cnt -= 1
        if not response:
            return '<end>'

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '继续' in p:
                return '<continue>'
            elif '结束' in p:
                return '<end>'
            else:
                return '<continue>'
        else:
            if 'continue' in p.lower():
                return '<continue>'
            elif 'end' in p.lower() or 'done' in p.lower() or 'finish' in p.lower():
                return '<end>'
            else:
                return '<continue>'

    def get_step_value(self, y):
        if y in self.value_cache:
            return self.value_cache[y]
        else:
            if self.value_method == 'gpt':
                value_prompt = self.single_value_wrap_gpt(self.question, y, self.lang)
            elif self.value_method == 'mistral':
                value_prompt = self.single_value_wrap_mistral(self.question, y)
            else:
                value_prompt = self.single_value_wrap(self.question, y, self.lang)
            cnt = 3
            response = []
            while not response and cnt:
                response = get_value(value_prompt, self.value_method, self.temperature, self.max_tokens, self.seed,
                                     self.max_length,
                                     self.truncation, self.do_sample, 128)
                cnt -= 1
            if not response:
                value = self.low
            else:
                p = ''
                for _ in response:
                    p = p + _ + ' '
                p = p.strip()
                try:
                    value = float(p)
                    value = max(self.low, min(self.high, value))
                except:
                    value = self.low
            self.value_cache[y] = value
            return value

    def search(self):
        self.set_limit_type()
        self.clear_cache()


        self.analyze_problem()


        node, finish, root = MCTS(self)


        if node and node.y:
            self.dynamic_strategy_selection(node.y)

        if node is None:
            if self.sample_value == 'full':
                end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
                value_samples = root.get_full_value_samples_vm(end_leaf_nodes)
                return value_samples
            else:
                return None
        else:
            return node.y



    def set_active_strategy(self, strategy_name):

        for strategy in self.strategies:
            if strategy.name == strategy_name:
                self.active_strategy = strategy
                return True
        return False

    def analyze_problem(self):

        question_lower = self.question.lower()


        strategy_scores = []
        for strategy in self.strategies:
            score = 0
            for keyword in strategy.keywords:
                if keyword.lower() in question_lower:
                    score += 1

        if strategy_scores:
            strategy_scores.sort(reverse=True)
            self.active_strategy = strategy_scores[0][1]

    def dynamic_strategy_selection(self, current_solution):


        if len(current_solution) < 200:
            if "pattern" in current_solution.lower() or "observe" in current_solution.lower():
                self.set_active_strategy("Induction Strategy")
            else:
                self.set_active_strategy("Deduction Strategy")


        elif len(current_solution) < 500:
            if "unclear" in current_solution.lower() or "not sure" in current_solution.lower():
                self.set_active_strategy("Abduction Strategy")


        else:
            if "therefore" in current_solution.lower() or "conclude" in current_solution.lower():
                self.set_active_strategy("Analogy Strategy")
