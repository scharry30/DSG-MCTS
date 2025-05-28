import re


def extract_summary_from_solution(solution):
    if not isinstance(solution, str):
        if isinstance(solution, dict) and 'summary' in solution:
            return solution['summary']
        return str(solution)

    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, solution)
    if boxed_matches:
        return boxed_matches[-1].strip()

    answer_pattern = r'(?:answer\s+is|答案\s*是|答案为)\s*[:\s]?\s*(.+?)(?:\.|$|\n)'
    answer_matches = re.findall(answer_pattern, solution, re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1].strip()

    therefore_pattern = r'(?:therefore|thus|所以|因此)\s*[,\s]?\s*(.+?)(?:\.|$|\n)'
    therefore_matches = re.findall(therefore_pattern, solution, re.IGNORECASE)
    if therefore_matches:
        return therefore_matches[-1].strip()

    number_pattern = r'(?:=\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    number_matches = re.findall(number_pattern, solution)
    if number_matches:
        return number_matches[-1].strip()

    lines = solution.strip().split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if non_empty_lines:
        return non_empty_lines[-1]

    return solution


def extract_equations(solution):
    if not isinstance(solution, str):
        return []

    equation_patterns = [
        r'([^=\n]+=[^=\n]+)',
        r'\\begin\{equation\}(.+?)\\end\{equation\}',
        r'\$(.+?=.+?)\$'
    ]

    equations = []
    for pattern in equation_patterns:
        matches = re.findall(pattern, solution)
        equations.extend([match.strip() for match in matches])

    return equations


def extract_reasoning_steps(solution):
    if not isinstance(solution, str):
        return []

    step_patterns = [
        r'(?:Step|步骤)\s*(\d+)\s*[:.]\s*(.+?)(?:\n|$)',
        r'(\d+)[.、]\s*(.+?)(?:\n|$)'
    ]

    for pattern in step_patterns:
        steps = re.findall(pattern, solution)
        if steps:
            return [(int(num), content.strip()) for num, content in steps]

    lines = solution.strip().split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    return [(i + 1, line) for i, line in enumerate(non_empty_lines) if len(line) > 10]
