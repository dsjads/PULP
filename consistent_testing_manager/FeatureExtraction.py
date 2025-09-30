import numpy as np


def calculate_cr(susp_in_variants, susp_scores_in_system, variant_stmt):
    # 修复1：检查susp_scores_in_system是否为字典，不是则直接返回0
    if not isinstance(susp_scores_in_system, dict):
        return 0
    high_susp_executed_stmt = set()
    all_executed_susp_stmt = {entry['id'] for entry in variant_stmt.values()}
    for item in susp_in_variants:
        high_susp_executed_stmt.update(susp_in_variants[item]["Executed"])
    total_stmts = len(all_executed_susp_stmt)
    average = sum(susp_scores_in_system.values())/len(susp_scores_in_system)
    susp_executed_score = 0
    for item in all_executed_susp_stmt:
        if  item in high_susp_executed_stmt:
            susp_executed_score += susp_scores_in_system[item]/average
    if total_stmts == 0:
        return 0
    return susp_executed_score/total_stmts


def normalize_dict_and_calculate_average(input_dict):
    # 1. 提取字典的键和值（确保键值顺序对应，Python 3.7+字典默认保留插入顺序）
    keys = list(input_dict.keys())
    values = list(input_dict.values())

    # 处理空字典边界情况
    if not keys:
        print("警告：输入字典为空，无法计算平均值")
        return None, {}  # 返回None（无平均值）和空字典（保持结构一致性）

    # 2. Min-Max归一化（0-1范围）
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        # 所有值相同，归一化后全为0.0
        normalized_values = [0.0 for _ in values]
    else:
        normalized_values = [(v - min_val) / (max_val - min_val) for v in values]

    # 3. 构建与输入结构一致的归一化字典（key不变，value替换为归一化后的值）
    normalized_dict = dict(zip(keys, normalized_values))

    # 4. 计算归一化后的平均值
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