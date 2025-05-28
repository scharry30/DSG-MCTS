import os
import pathlib
import random
from CoT.task import CoT_Task
from ToT.task import ToT_Task
from MCTS.task import MCTS_Task
import argparse
from utils.visualize import visualize
from utils.json_operator import *
from utils.verify_answer import *
from utils.self_consistency import get_consistency_output_scibench


def run(arguments):
    print('-' * 30, 'Begin testing', '-' * 30, '\n')
    file = f'data/{arguments.task_name}/{arguments.file}.json'
    try:
        data_list = read_json(file)
        data_len = len(data_list)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return
    assert data_len > 0, "Data list is empty!\n"
    assert 'content' in data_list[0].keys() and 'answer' in data_list[
        0].keys(), "Key error, Make sure json object contain correct keys!\n"


    if arguments.seed is not None:
        random.seed(arguments.seed)

    output_list = []
    correct_count = 0
    for i in range(data_len):

        print(f'Begin to solve the problem {i + 1}...\n')
        data = data_list[i]['content']
        answer = data_list[i]['answer']

        if arguments.mode == 'cot':
            Task = CoT_Task(data, arguments.propose_method, arguments.value_method, arguments.temperature,
                            evaluate=arguments.evaluate, lang=arguments.lang)
            if arguments.consistency:
                outputs = []
                for cnt in range(3):
                    output = Task.run()
                    outputs.append(output)
                output = get_consistency_output_scibench(outputs)
            else:
                output = Task.run()

        elif arguments.mode == 'tot':
            Task = ToT_Task(data, arguments.propose_method, arguments.value_method, arguments.algorithm,
                            arguments.branch, arguments.select_branch, arguments.max_depth, arguments.end_gate,
                            arguments.select_method, arguments.temperature, use_case_prompt=arguments.use_case_prompt,
                            low=arguments.low, high=arguments.high, evaluate=arguments.evaluate, lang=arguments.lang)
            output, root = Task.run()
            if arguments.visualize:
                visualize(root, Task, arguments.task_name, arguments.file, i + 1)

        else:
            Task = MCTS_Task(data, arguments.propose_method, arguments.value_method, arguments.branch,
                             arguments.end_gate,
                             arguments.roll_policy, arguments.roll_branch, arguments.roll_forward_steps,
                             arguments.time_limit,
                             arguments.iteration_limit, arguments.exploration_constant, arguments.alpha, arguments.inf,
                             arguments.temperature, max_tokens=arguments.max_tokens, seed=arguments.seed,
                             max_length=arguments.max_length, truncation=arguments.truncation,
                             do_sample=arguments.do_sample, max_new_tokens=arguments.max_new_tokens,
                             use_case_prompt=arguments.use_case_prompt, use_reflection=arguments.use_reflection,
                             low=arguments.low, high=arguments.high, evaluate=arguments.evaluate,
                             sample_value=arguments.sample_value, answer=answer,
                             verify_method=arguments.verify_method, lang=arguments.lang)


            if arguments.strategy_selection == 'auto':
                Task.analyze_problem()
            elif arguments.strategy_selection in ['deduction', 'induction', 'abduction', 'analogy']:
                for strategy in Task.strategies:
                    if strategy.name.lower().startswith(arguments.strategy_selection.lower()):
                        Task.active_strategy = strategy
                        strategy.reset_actions()  # 重置策略的动作序列
                        break


            Task.strategy_weight = arguments.strategy_weight
            Task.diversity_weight = arguments.diversity_weight

            result = Task.search()


            if isinstance(result, dict) and 'solution' in result:
                output = result
            else:
                output = {'solution': result}


            if arguments.visualize and hasattr(Task, 'root'):
                visualize(Task.root, Task, arguments.task_name, arguments.file, i + 1)

        if arguments.evaluate:

            if not isinstance(output, dict):
                output = {'solution': output, 'summary': output}
            elif 'summary' not in output and 'solution' in output:

                output['summary'] = extract_summary_from_solution(output['solution'])


            result = verify_float(answer, output['summary'])
            output.update({'answer': answer, 'accurate': result})

            if result:
                print(f'The answer of problem {i + 1} is correct.\n')
                correct_count += 1
            else:
                print(f'The answer of problem {i + 1} is wrong.\n')

        print(f'The solution to problem {i + 1} is complete.\n')


        if arguments.mode == 'mcts' and hasattr(Task, 'strategy_usage'):
            print("Strategy usage statistics:")
            for strategy, count in Task.strategy_usage.items():
                print(f"- {strategy}: {count} times")
            print()


        base_dir = os.getcwd()

        if arguments.mode == 'mcts' and hasattr(Task, 'active_strategy'):
            strategy_name = Task.active_strategy.name.split()[0].lower()
            output_dir = pathlib.Path(f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}')
            output_file = f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}/{Task.propose_method}_{Task.value_method}_{strategy_name}.json'
        else:
            output_dir = pathlib.Path(f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}')
            output_file = f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}/{Task.propose_method}_{Task.value_method}.json'

        output_list.append(output)
        pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
        dump_json(output_file, output_list)

    print('_' * 60)

    if arguments.evaluate:
        print(f'Test accuracy:{correct_count / data_len}\n')
        print(f'Correct number of problems:{correct_count}\nTotal number of questions:{data_len}\n')
    print('_' * 60)


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='scibench')
    base_args.add_argument('--file', type=str, default='thermo_standardized')  # json
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'local', 'mistral'],
                           default='glm')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local', 'mistral'], default='local')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='mcts')
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--max_tokens', type=int, default=2048)
    base_args.add_argument('--seed', type=int, default=170)
    base_args.add_argument('--max_length', type=int, default=2048)
    base_args.add_argument('--truncation', type=bool, default=True)
    base_args.add_argument('--do_sample', type=bool, default=True)
    base_args.add_argument('--max_new_tokens', type=int, default=256)
    base_args.add_argument('--time_limit', type=int, default=None)
    base_args.add_argument('--iteration_limit', type=int, default=100)
    base_args.add_argument('--roll_policy', type=str, choices=['random', 'greedy'], default='greedy')
    base_args.add_argument('--exploration_constant', type=float, default=0.4)
    base_args.add_argument('--roll_forward_steps', type=int, default=2)
    base_args.add_argument('--end_gate', type=float, default=0.9)  # End threshold
    base_args.add_argument('--branch', type=int, default=3)
    base_args.add_argument('--roll_branch', type=int, default=1)
    base_args.add_argument('--inf', type=float, default=0.8)
    base_args.add_argument('--evaluate', type=str,
                           default='scibench')  # Whether to evaluate (empty means no evaluation)
    base_args.add_argument('--alpha', type=float, default=0.5)
    base_args.add_argument('--visualize', type=bool, default=False)  # visualization
    base_args.add_argument('--use_case_prompt', type=bool, default=False)  # Use sample prompts
    base_args.add_argument('--use_reflection', type=str, choices=['simple', 'common'],
                           default='simple')  # Use reflective mode
    base_args.add_argument('--low', type=float, default=0)
    base_args.add_argument('--high', type=float, default=1)
    base_args.add_argument('--algorithm', type=str, choices=['dfs', 'bfs'], default='dfs')
    base_args.add_argument('--select_branch', type=int, default=2)
    base_args.add_argument('--max_depth', type=int, default=8)
    base_args.add_argument('--select_method', type=str, choices=['greedy', 'sample'], default='greedy')
    base_args.add_argument('--consistency', type=bool, default=True)
    base_args.add_argument('--sample_value', type=str, choices=['simple', 'full'], default='simple')


    base_args.add_argument('--strategy_selection', type=str,
                           choices=['auto', 'deduction', 'induction', 'abduction', 'analogy'], default='auto',
                           help='Strategy selection method: auto or specific strategy')
    base_args.add_argument('--strategy_weight', type=float, default=0.5,
                           help='Weight for strategy influence in search')
    base_args.add_argument('--diversity_weight', type=float, default=0.2,
                           help='Weight for diversity in search')
    base_args.add_argument('--verify_method', type=str, choices=['string', 'llm'], default='string',
                           help='Method for answer verification')
    base_args.add_argument('--lang', type=str, choices=['en', 'zh'], default='en',
                           help='Language for prompts and responses')

    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    run(args)
