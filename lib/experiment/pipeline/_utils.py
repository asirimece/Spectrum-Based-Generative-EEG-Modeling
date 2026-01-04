from typing import List


def get_best_values_from_history(history: List) -> dict:
    metrics = {}
    if history is None:
        return metrics
    for entry in history:
        for key in entry:
            if key.endswith('_best') and isinstance(entry[key], bool) and entry[key] is True:
                metric_name = key.split('_best')[0]
                metrics[metric_name] = entry[metric_name]
    return metrics
