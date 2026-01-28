import statistics

import numpy as np

# susp in variants,记录失败产品各失败测试用例的执行路径
def calculate_cr(susp_in_variants, susp_scores_in_system, variant_stmt):
    if not isinstance(susp_scores_in_system, dict):
        return 0
    susp_scores = list(susp_scores_in_system.values())
    median = statistics.median(susp_scores)

    high_susp_executed_stmt = {
        key for key,value in susp_scores_in_system.items()
        if value > median
    }
    all_executed_susp_stmt = {entry['id'] for entry in variant_stmt.values()}
    for item in susp_in_variants:
        high_susp_executed_stmt.update(susp_in_variants[item]["Executed"])
    total_stmts = len(all_executed_susp_stmt)
    # average = sum(susp_scores_in_system.values())/len(susp_scores_in_system)
    executed_susp_number = 0
    for item in all_executed_susp_stmt:
        if  item in high_susp_executed_stmt:
            # susp_executed_score += susp_scores_in_system[item]/average
            executed_susp_number += 1
    if total_stmts == 0:
        return 0
    return executed_susp_number/total_stmts


def normalize_dict_and_calculate_average(input_dict):
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    if not keys:
        print("警告：输入字典为空，无法计算平均值")
        return None, {}
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        normalized_values = [0.0 for _ in values]
    else:
        normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
    normalized_dict = dict(zip(keys, normalized_values))
    average = sum(normalized_values) / len(normalized_values)
    return average, normalized_dict

def convert_to_dict(passing_variant_stmt):
    stmt_ids = {}
    for item in passing_variant_stmt:
        tmp = passing_variant_stmt[item]
        if tmp["id"] not in stmt_ids:
            stmt_ids[tmp["id"]] = {}
            stmt_ids[tmp["id"]]["tested"] = tmp["tested"]
    return stmt_ids