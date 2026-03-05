from __future__ import annotations

import math
import statistics
from collections import defaultdict

from eval.benchmark_v2.types import CounterfactualTriple, RawScoreRecord


def _valid_pairs(human_scores: list[float], model_scores: list[float | None]) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for human_score, model_score in zip(human_scores, model_scores):
        if model_score is None:
            continue
        x_values.append(human_score)
        y_values.append(model_score)
    return x_values, y_values


def mae(human_scores: list[float], model_scores: list[float | None]) -> float:
    x_values, y_values = _valid_pairs(human_scores, model_scores)
    if not x_values:
        return math.nan
    errors = [abs(x - y) for x, y in zip(x_values, y_values)]
    return sum(errors) / len(errors)


def rmse(human_scores: list[float], model_scores: list[float | None]) -> float:
    x_values, y_values = _valid_pairs(human_scores, model_scores)
    if not x_values:
        return math.nan
    squared_errors = [(x - y) ** 2 for x, y in zip(x_values, y_values)]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def pearson(human_scores: list[float], model_scores: list[float | None]) -> float:
    x_values, y_values = _valid_pairs(human_scores, model_scores)
    if len(x_values) < 2:
        return math.nan
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    if x_var == 0 or y_var == 0:
        return math.nan
    return numerator / math.sqrt(x_var * y_var)


def _ranks(values: list[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(values):
        next_index = index + 1
        while next_index < len(values) and pairs[next_index][1] == pairs[index][1]:
            next_index += 1
        avg_rank = (index + next_index - 1) / 2 + 1
        for offset in range(index, next_index):
            original_index = pairs[offset][0]
            ranks[original_index] = avg_rank
        index = next_index
    return ranks


def spearman(human_scores: list[float], model_scores: list[float | None]) -> float:
    x_values, y_values = _valid_pairs(human_scores, model_scores)
    if len(x_values) < 2:
        return math.nan
    return pearson(_ranks(x_values), _ranks(y_values))


def ccc(human_scores: list[float], model_scores: list[float | None]) -> float:
    x_values, y_values = _valid_pairs(human_scores, model_scores)
    if len(x_values) < 2:
        return math.nan
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    x_var = statistics.pvariance(x_values)
    y_var = statistics.pvariance(y_values)
    xy_cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / len(x_values)
    denominator = x_var + y_var + (x_mean - y_mean) ** 2
    if denominator == 0:
        return math.nan
    return (2 * xy_cov) / denominator


def bucket_mae(
    human_scores: list[float],
    model_scores: list[float | None],
    buckets: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
    bins = buckets or [(0, 24), (25, 49), (50, 74), (75, 100)]
    per_bucket_errors: dict[str, list[float]] = {f"{low}-{high}": [] for low, high in bins}
    for human_score, model_score in zip(human_scores, model_scores):
        if model_score is None:
            continue
        for low, high in bins:
            if low <= human_score <= high:
                per_bucket_errors[f"{low}-{high}"].append(abs(human_score - model_score))
                break
    output: dict[str, float] = {}
    for bucket_name, errors in per_bucket_errors.items():
        output[bucket_name] = statistics.fmean(errors) if errors else math.nan
    return output


def mean_test_retest_std(records: list[RawScoreRecord]) -> float:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.score is None:
            continue
        key = (record.model, record.sample_id)
        grouped[key].append(record.score)
    std_values = [statistics.pstdev(scores) for scores in grouped.values() if len(scores) > 1]
    if not std_values:
        return math.nan
    return statistics.fmean(std_values)


def mean_prompt_paraphrase_std(records: list[RawScoreRecord]) -> float:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for record in records:
        if record.score is None:
            continue
        key = (record.model, record.sample_id, record.repeat_id)
        grouped[key].append(record.score)
    std_values = [statistics.pstdev(scores) for scores in grouped.values() if len(scores) > 1]
    if not std_values:
        return math.nan
    return statistics.fmean(std_values)


def icc2_1(records: list[RawScoreRecord], model_name: str) -> float:
    per_sample: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.model != model_name or record.score is None:
            continue
        per_sample[record.sample_id].append(record.score)
    matrix = [scores for scores in per_sample.values() if len(scores) > 1]
    if len(matrix) < 2:
        return math.nan
    row_count = len(matrix)
    col_count = min(len(row) for row in matrix)
    trimmed = [row[:col_count] for row in matrix]
    grand_mean = statistics.fmean([value for row in trimmed for value in row])
    row_means = [statistics.fmean(row) for row in trimmed]
    col_means = [statistics.fmean([row[col_index] for row in trimmed]) for col_index in range(col_count)]
    msr = (col_count * sum((row_mean - grand_mean) ** 2 for row_mean in row_means)) / (row_count - 1)
    msc = (row_count * sum((col_mean - grand_mean) ** 2 for col_mean in col_means)) / (col_count - 1)
    mse_numerator = 0.0
    for row_index, row in enumerate(trimmed):
        for col_index, value in enumerate(row):
            mse_numerator += (value - row_means[row_index] - col_means[col_index] + grand_mean) ** 2
    mse = mse_numerator / ((row_count - 1) * (col_count - 1))
    denominator = msr + (col_count - 1) * mse + (col_count * (msc - mse) / row_count)
    if denominator == 0:
        return math.nan
    return (msr - mse) / denominator


def counterfactual_order_accuracy(
    triples: list[CounterfactualTriple],
    scores_by_sample: dict[str, float],
) -> float:
    if not triples:
        return math.nan
    correct_count = 0
    valid_count = 0
    for triple in triples:
        light_score = scores_by_sample.get(triple.light_id)
        heavy_score = scores_by_sample.get(triple.heavy_id)
        if light_score is None or heavy_score is None:
            continue
        valid_count += 1
        if triple.expected_order == "heavy_gt_light" and heavy_score > light_score:
            correct_count += 1
        elif triple.expected_order == "heavy_gte_light" and heavy_score >= light_score:
            correct_count += 1
    if valid_count == 0:
        return math.nan
    return correct_count / valid_count

