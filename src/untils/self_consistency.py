from collections import Counter
from utils.verify_answer import normalize_number
import re


def extract_numbers_from_text(text):
    if not isinstance(text, str):
        text = str(text)

    number_pattern = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    return re.findall(number_pattern, text)


def get_consistency_output_scibench(outputs):
    if not outputs:
        return {}

    if len(outputs) == 1:
        return outputs[0]

    summaries = []
    for output in outputs:
        if isinstance(output, dict) and 'summary' in output:
            summaries.append(output['summary'])
        else:
            summaries.append(str(output))

    numbers = []
    for summary in summaries:
        extracted = extract_numbers_from_text(summary)
        if extracted:
            numbers.append(normalize_number(extracted[-1]))

    if not numbers:
        return outputs[0]

    number_counts = Counter(numbers)
    most_common_number = number_counts.most_common(1)[0][0]

    for i, summary in enumerate(summaries):
        if most_common_number in summary:
            return outputs[i]

    return outputs[0]


def get_consistency_output_general(outputs, key='solution'):
    if not outputs:
        return {}

    if len(outputs) == 1:
        return outputs[0]

    values = []
    for output in outputs:
        if isinstance(output, dict) and key in output:
            values.append(output[key])
        else:
            values.append(str(output))

    value_counts = Counter(values)
    most_common_value = value_counts.most_common(1)[0][0]

    for i, value in enumerate(values):
        if value == most_common_value:
            return outputs[i]

    return outputs[0]
