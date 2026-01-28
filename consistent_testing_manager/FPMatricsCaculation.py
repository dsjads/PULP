import csv
import json
from collections import defaultdict
from multiprocessing import Pool

import pandas
from sklearn import preprocessing
from FileManager import *
from consistent_testing_manager.DDU import ddu
from consistent_testing_manager.FeatureExtraction import calculate_cr

from spectrum_manager.SpectrumReader import get_stm_ids_per_variant, similar_path, \
    get_suspicious_space_consistent_version, get_infor_for_sbfl, get_passing_executions_in_a_variant, \
    get_failings_executions, get_passing_executions, get_all_stm_ids
from ranking.RankingManager import get_set_of_stms, sbfl_ranking
from spectrum_manager.Spectrum_Expression import OCHIAI, BARINEL, OP2, DSTAR, RUSSELL_RAO, JACCARD, TARANTULA, GP02, \
    GP03, GP19
from suspicious_statements_manager.SlicingManager import do_slice_all_statements

BACKWARD_SLICING_TYPE = "Backward"
FORWARD_SLICING_TYPE = "Forward"
BOTH_FB_SLICING_TYPE = "Both"

TRUE_PASSING = "TP"
FALSE_PASSING = "FP"
FAILING = "F"

VARIANT_NAME = 'VARIANT'
LABEL = 'LABEL'
TRANSFORMED_FP = 'FP transformed from F'
DDU = "DDU"

buggy_statement_containing_possibility = "bscp"
executed_susp_stmt_in_passing_variant = "executed_susp_stmt_in_passing_variant"
code_coverage = "code_coverage"

incorrectness_verifiability = "incorrectness_verifiability"
correctness_reflectability = "correctness_reflectability"

bug_involving_statements = "bug_involving_statements"


CLASSIFY_ATTRIBUTES = [DDU, code_coverage,
                       incorrectness_verifiability,
                       correctness_reflectability,
                       buggy_statement_containing_possibility,
                       bug_involving_statements]

FIELDS = [VARIANT_NAME, LABEL, DDU,
          code_coverage,
          incorrectness_verifiability,
          correctness_reflectability,
          buggy_statement_containing_possibility,
          bug_involving_statements]


def get_variants_and_labels(mutated_project_dir, label_file):
    variants = {}
    label_file = join_path(mutated_project_dir, label_file)
    with open(label_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            variants[row[VARIANT_NAME]] = {LABEL: row[LABEL], TRANSFORMED_FP: row[TRANSFORMED_FP]}
    return variants


def write_dict_to_file(file_name, data, fieldnames):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            tmp = {}
            for f in fieldnames:
                if fieldnames.index(f) == 0:
                    tmp[f] = item
                else:
                    tmp[f] = data[item][f]
            writer.writerow(tmp)

def write_dict_to_file2(file_name, data):
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        # 处理空数据情况
        if not data:
            return

        # 获取第一个item的features，确定基础列名
        first_item_key = next(iter(data.keys()))  # 第一个item的key
        first_item_features = data[first_item_key]  # 第一个item的features

        # 构造列名：key列（最前面） + features中的所有键（保持原有顺序）
        fieldnames = ['model'] + list(first_item_features.keys())

        # 初始化DictWriter
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头

        # 遍历每个item写入数据
        for item_key, features in data.items():
            # 构建行数据：先加入key值，再加入features中的键值对
            row_data = {'model': item_key}  # key列对应的值
            # 补充features中的所有字段（确保顺序与fieldnames一致）
            for field in fieldnames[1:]:  # 跳过第一个'key'字段
                row_data[field] = features.get(field, '')  # 缺失字段用空字符串填充
            writer.writerow(row_data)

def get_labeled_failing_variants(project_dir, label_file):
    failing_variants = get_failing_variants(project_dir)
    variants_and_labels = get_variants_and_labels(project_dir, label_file)
    for item in failing_variants:
        if variants_and_labels[item][LABEL] != FAILING:
            failing_variants.remove(item)
    return failing_variants


def get_stmts_id_in_passing_variants(project_dir, failing_variants):
    passing_variants_stmts = {}
    variants_dir = get_variants_dir(project_dir)
    variants_list = list_dir(variants_dir)
    for variant in variants_list:
        if variant not in failing_variants:
            variant_dir = join_path(variants_dir, variant)
            variant_stmts = get_stm_ids_per_variant(variant_dir, fp_variant=True)
            passing_variants_stmts[variant] = variant_stmts
    return passing_variants_stmts


def convert_to_dict(passing_variant_stmt):
    stmt_ids = {}
    for item in passing_variant_stmt:
        tmp = passing_variant_stmt[item]
        if tmp["id"] not in stmt_ids:
            stmt_ids[tmp["id"]] = {}
            stmt_ids[tmp["id"]]["tested"] = tmp["tested"]
    return stmt_ids


def convert_execution_to_set(execution):
    execution_set = set()
    for item in execution:
        execution_set.add(item["id"])
    return execution_set


def exist_path(path, list_paths, threshold):
    for p in list_paths:
        if similar_path(path, list_paths[p], threshold):
            return True
    return False

# 修改
def check_suspicious_stmts_in_passing_variants(failing_executions, passing_variant_stmt):
    passing_variant_stmt_dict = convert_to_dict(passing_variant_stmt)
    suspicious_in_passing_variant = {}
    # failing executions： {model_m_1 : {test1_coverage:[], test2_coverage:[]}, model_m_2: {test1_coverage:[], test3_coverage:[]}} ...
    for variant_name in failing_executions:
        for test in failing_executions[variant_name]:
            execution = failing_executions[variant_name][test]
            D1 = []
            D2 = []
            for item in execution:
                if item["id"] in passing_variant_stmt_dict:
                    if passing_variant_stmt_dict[item["id"]]["tested"] == 1:
                        D1.append(item["id"])
                    else:
                        D2.append(item["id"])
            suspicious_in_passing_variant[variant_name + "__" + test] = {"Executed": D1, "Not Executed": D2}

    return suspicious_in_passing_variant


# 计算noncov的方法
def check_executed_susp_stmt_vs_susp_stmt_in_passing_variant(susp_in_passing_variant):
    executed_suspicious_stmt = set()
    not_executed_suspcious_stmt = set()
    for item in susp_in_passing_variant:
        executed_suspicious_stmt.update(susp_in_passing_variant[item]["Executed"])
        not_executed_suspcious_stmt.update(susp_in_passing_variant[item]["Not Executed"])
    for item in executed_suspicious_stmt:
        not_executed_suspcious_stmt.discard(item)

    total_suspicious_stmts = len(executed_suspicious_stmt) + len(not_executed_suspcious_stmt)
    if total_suspicious_stmts == 0:
        return 0
    return len(not_executed_suspcious_stmt)/total_suspicious_stmts


def jaccard_similarity(set1, set2):
    interaction = len(set1.intersection(set2))
    union = len(set1) + len(set2) - interaction
    if union == 0:
        return 0
    return float(interaction) / union


def check_incorrectness_verifiability(executions_in_failing_products, execution_in_a_passing_product,
                                      susp_in_passing_product,
                                      threshold):
    num_failed_executions_contained_in_passing_product = 0
    num_failed_executions_tested_by_passing_product = 0
    # 遍历当前product和其他failing product每一个测试用例
    for item in susp_in_passing_product:
        var_name = item.split("__")[0]
        test_id = item.split("__")[1]
        # 取出该failing test的总路径
        failed_execution = executions_in_failing_products[var_name][test_id]
        # susp_set_len为当前product和当前遍历的failing test重合的语句数量（不一定执行了）
        susp_set_len = len(susp_in_passing_product[item]["Executed"]) + len(
            susp_in_passing_product[item]["Not Executed"])
        # 如果在当前遍历的failing test上，当前product具有的语句占失败测试用例的一定比率之上
        if susp_set_len / len(failed_execution) > threshold:
            # 当前product上存在失败测试路径的数量
            num_failed_executions_contained_in_passing_product += 1
            # 失败测试路径被product的passing路径执行的数量
            if exist_path(failed_execution, execution_in_a_passing_product, threshold):
                num_failed_executions_tested_by_passing_product += 1

    if num_failed_executions_contained_in_passing_product == 0:
        return 0
    else:
        return 1 - num_failed_executions_tested_by_passing_product / num_failed_executions_contained_in_passing_product

def check_similarity_score(executions_in_failing_products, susp_in_passing_product, susp_scores_in_system):
    if len(susp_in_passing_product)==0:
        return 0
    res = 0
    for item in susp_in_passing_product:
        var_name = item.split("__")[0]
        test_id = item.split("__")[1]
        # 取出该failing test的总路径
        failed_execution = executions_in_failing_products[var_name][test_id]
        susp_list_in_failing_test = [item['id'] for item in failed_execution]
        total_score = get_susp_score(susp_list_in_failing_test, susp_scores_in_system)
        susp_list_in_current_product = susp_in_passing_product[item]["Executed"] + susp_in_passing_product[item]["Not Executed"]
        susp_set_score = get_susp_score(susp_list_in_current_product, susp_scores_in_system)
        res = res + susp_set_score/total_score
    return res/len(susp_in_passing_product)

def get_susp_score(susp_in_passing_product, susp_scores_in_system):
    res = 0
    for item in susp_in_passing_product:
        res = res + susp_scores_in_system[item]
    return res

# 改变threshold,可以获取到多组特征
def check_correctness_reflectability(failed_executions_in_failing_products,
                                     passed_executions_in_failing_products,
                                     passed_executions_in_passing_product, susp_in_passing_product,
                                     threshold):
    failing_set_in_failing_variants = []
    for variant in failed_executions_in_failing_products:
        for path in failed_executions_in_failing_products[variant]:
            set_tmp = convert_execution_to_set(failed_executions_in_failing_products[variant][path])
            failing_set_in_failing_variants.append(set_tmp)

    passing_set_in_failing_variants = []
    for variant in passed_executions_in_failing_products:
        for path in passed_executions_in_failing_products[variant]:
            set_tmp = convert_execution_to_set(passed_executions_in_failing_products[variant][path])
            passing_set_in_failing_variants.append(set_tmp)

    coincidentally_passed_tests = []
    for passed in passing_set_in_failing_variants:
        for failed in failing_set_in_failing_variants:
            if jaccard_similarity(passed, failed) > threshold:
                if passed not in coincidentally_passed_tests:
                    coincidentally_passed_tests.append(passed)

    count = 0
    for path in passed_executions_in_passing_product:
        set_tmp = convert_execution_to_set(passed_executions_in_passing_product[path])
        for p_temp in coincidentally_passed_tests:
            if jaccard_similarity(set_tmp, p_temp) > threshold:
                count += 1
                break
    return count / len(passed_executions_in_passing_product)

# 可疑语句排序
def ranking_suspicious_stmts(project_dir, failing_variants):
    search_spaces = get_suspicious_space_consistent_version(project_dir, failing_variants, 0.0, "")
    variants = list_dir(get_variants_dir(project_dir))
    fp_variants = []
    for v in variants:
        if v not in failing_variants:
            fp_variants.append(v)
    stm_info_for_sbfl, total_passed_tests, total_failed_tests = get_infor_for_sbfl(
        project_dir, failing_variants=failing_variants, fp_variants=fp_variants,
        spectrum_coverage_prefix="",
        coverage_rate=0.0)
    all_stms_f_products_set = get_set_of_stms(search_spaces)
    full_ranked_list = sbfl_ranking(stm_info_for_sbfl, total_failed_tests, total_passed_tests,
                                    all_stms_f_products_set,
                                    [OCHIAI])
                                    # [JACCARD, TARANTULA, OCHIAI, OP2, DSTAR, BARINEL, RUSSELL_RAO, GP02, GP03, GP19])
    # full_ranked_map_list = []
    # for ranked_list in full_ranked_list.values():
    #     temp_ranked_map = {}
    #     for (stmt, score, v) in ranked_list:
    #         temp_ranked_map[stmt] = score
    #     full_ranked_map_list.append(temp_ranked_map)
    # return full_ranked_map_list
    op2_ranked_list = {}
    for (stmt, score, v) in full_ranked_list[OCHIAI]:
        op2_ranked_list[stmt] = score
    return op2_ranked_list


def check_total_susp_scores_in_passing_variant(susp_scores, passing_variant_stmt):
    sum_scores = 0
    num_of_stm = 0
    for stmt in passing_variant_stmt:
        tmp = passing_variant_stmt[stmt]["id"]
        if tmp in susp_scores:
            sum_scores += susp_scores[tmp]
            num_of_stm += 1
    if num_of_stm == 0:
        return 0
    return sum_scores / num_of_stm


def get_dependencies(slicing_file_dir):
    slicing_file = open(slicing_file_dir, "r")
    slicing_content = slicing_file.readline()
    slicies = json.loads(slicing_content)
    return slicies

# 修改函数，加入可疑值加权
def check_dependencies_by_slicing_type(similarities, susp_stmt, fv_slicies, pv_slicies, slicing_type):
    if susp_stmt in fv_slicies:
        if susp_stmt in pv_slicies:
            simi_score = jaccard_similarity(set(pv_slicies[susp_stmt]), set(fv_slicies[susp_stmt]))
            if susp_stmt not in similarities:
                similarities[susp_stmt] = {}
                similarities[susp_stmt]["Backward"] = 0
                similarities[susp_stmt]["Forward"] = 0
                similarities[susp_stmt]["Both"] = 0

            if simi_score > similarities[susp_stmt][slicing_type]:
                similarities[susp_stmt][slicing_type] = simi_score
    return similarities

def check_dependencies_by_slicing_type_with_susp_score(similarities, susp_stmt, fv_slicies, pv_slicies, slicing_type, susp_scores_in_system):
    if susp_scores_in_system[max(susp_scores_in_system)] == 0:
        similarities[susp_stmt] = {}
        similarities[susp_stmt]["Backward"] = 0
        similarities[susp_stmt]["Forward"] = 0
        similarities[susp_stmt]["Both"] = 0
    elif susp_stmt in fv_slicies:
        if susp_stmt in pv_slicies:
            simi_score = jaccard_similarity(set(pv_slicies[susp_stmt]), set(fv_slicies[susp_stmt]))*(susp_scores_in_system[susp_stmt]/susp_scores_in_system[max(susp_scores_in_system)])
            if susp_stmt not in similarities:
                similarities[susp_stmt] = {}
                similarities[susp_stmt]["Backward"] = 0
                similarities[susp_stmt]["Forward"] = 0
                similarities[susp_stmt]["Both"] = 0
            if simi_score > similarities[susp_stmt][slicing_type]:
                similarities[susp_stmt][slicing_type] = simi_score
    return similarities


def aggregate_similarity_by_avg(similarity):
    fw = 0
    bw = 0
    both = 0
    for stmt in similarity:
        fw += similarity[stmt]["Forward"]
        bw += similarity[stmt]["Backward"]
        both += similarity[stmt]["Both"]

    if len(similarity) == 0:
        return {"Forward": 0, "Backward": 0, "Both": 0}
    return {"Forward": fw / len(similarity), "Backward": bw / len(similarity), "Both": both / len(similarity)}


def concat_slicies(forward_slicies, backward_slicies):
    both_slicies = defaultdict(dict)
    for item in forward_slicies:
        both_slicies[item] = set(forward_slicies[item])

    for item in backward_slicies:
        if item not in both_slicies:
            both_slicies[item] = set(backward_slicies[item])
        else:
            both_slicies[item].update(backward_slicies[item])
    return both_slicies

def check_dependencies(variants_folder_dir, passing_variant, failed_executions_in_failing_products):
    passing_variant_dir = join_path(variants_folder_dir, passing_variant)
    pv_forward_file = get_forward_slicing_file(passing_variant_dir)
    pv_forward_slicies = get_dependencies(pv_forward_file)

    pv_backward_file = get_backward_slicing_file(passing_variant_dir)
    pv_backward_slicies = get_dependencies(pv_backward_file)

    pv_both_slicies = concat_slicies(pv_forward_slicies, pv_backward_slicies)

    similarities = {}
    for fv in failed_executions_in_failing_products:
        fv_dir = join_path(variants_folder_dir, fv)
        fv_forward_file = get_forward_slicing_file(fv_dir)
        fv_forward_slicies = get_dependencies(fv_forward_file)
        fv_backward_file = get_backward_slicing_file(fv_dir)
        fv_backward_slicies = get_dependencies(fv_backward_file)
        fv_both_slicies = concat_slicies(fv_forward_slicies, fv_backward_slicies)

        for test in failed_executions_in_failing_products[fv]:
            for item in failed_executions_in_failing_products[fv][test]:
                susp_stmt = item["id"]
                similarities = check_dependencies_by_slicing_type(similarities, susp_stmt, fv_forward_slicies,
                                                                  pv_forward_slicies, FORWARD_SLICING_TYPE)
                similarities = check_dependencies_by_slicing_type(similarities, susp_stmt, fv_backward_slicies,
                                                                  pv_backward_slicies, BACKWARD_SLICING_TYPE)
                similarities = check_dependencies_by_slicing_type(similarities, susp_stmt, fv_both_slicies,
                                                                  pv_both_slicies, BOTH_FB_SLICING_TYPE)
    return aggregate_similarity_by_avg(similarities)

def check_dependencies_with_susp_score(variants_folder_dir, passing_variant, failed_executions_in_failing_products, susp_scores_in_system):
    passing_variant_dir = join_path(variants_folder_dir, passing_variant)
    pv_forward_file = get_forward_slicing_file(passing_variant_dir)
    pv_forward_slicies = get_dependencies(pv_forward_file)

    pv_backward_file = get_backward_slicing_file(passing_variant_dir)
    pv_backward_slicies = get_dependencies(pv_backward_file)

    pv_both_slicies = concat_slicies(pv_forward_slicies, pv_backward_slicies)
    similarities = {}
    for fv in failed_executions_in_failing_products:
        fv_dir = join_path(variants_folder_dir, fv)
        fv_forward_file = get_forward_slicing_file(fv_dir)
        fv_forward_slicies = get_dependencies(fv_forward_file)
        fv_backward_file = get_backward_slicing_file(fv_dir)
        fv_backward_slicies = get_dependencies(fv_backward_file)
        fv_both_slicies = concat_slicies(fv_forward_slicies, fv_backward_slicies)

        for test in failed_executions_in_failing_products[fv]:
            for item in failed_executions_in_failing_products[fv][test]:
                susp_stmt = item["id"]
                similarities = check_dependencies_by_slicing_type_with_susp_score(similarities, susp_stmt, fv_forward_slicies,
                                                                                  pv_forward_slicies, FORWARD_SLICING_TYPE, susp_scores_in_system)
                similarities = check_dependencies_by_slicing_type_with_susp_score(similarities, susp_stmt, fv_backward_slicies,
                                                                                  pv_backward_slicies,BACKWARD_SLICING_TYPE, susp_scores_in_system)
                similarities = check_dependencies_by_slicing_type_with_susp_score(similarities, susp_stmt, fv_both_slicies,
                                                                                  pv_both_slicies, BOTH_FB_SLICING_TYPE, susp_scores_in_system)
    return aggregate_similarity_by_avg(similarities)

def normalization(FIELDS, project_dir, attribute_file, attribute_normalized_file):
    consistent_testing_info_file_path = join_path(project_dir, attribute_file)
    consistent_testing_info_normalized_file = join_path(project_dir, attribute_normalized_file)
    data = pandas.read_csv(consistent_testing_info_file_path)
    variants = data[FIELDS[0]]
    labels = data[FIELDS[1]]
    x = data[FIELDS[2:]].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pandas.DataFrame(x_scaled)
    data.columns = FIELDS[2:]
    data.insert(loc=0, column=FIELDS[1], value=labels)
    data.insert(loc=0, column=FIELDS[0], value=variants)
    data.to_csv(consistent_testing_info_normalized_file)

def normalization2(project_dir, attribute_file, attribute_normalized_file):
    consistent_testing_info_file_path = join_path(project_dir, attribute_file)
    consistent_testing_info_normalized_file = join_path(project_dir, attribute_normalized_file)
    data = pandas.read_csv(consistent_testing_info_file_path)

    # 确定需要归一化的列（排除'model'和'LABEL'）
    columns_to_normalize = [col for col in data.columns if col not in ['model', 'LABEL']]

    # 对每列进行最小-最大归一化 (Min-Max Normalization)
    for col in columns_to_normalize:
        # 获取当前列的最小值和最大值
        col_min = data[col].min()
        col_max = data[col].max()

        # 处理极端情况：如果所有值都相同（避免除以0）
        if col_max == col_min:
            data[col] = 0.0  # 全部归一化为0
        else:
            # 归一化公式：(x - min) / (max - min)，将值缩放到[0, 1]区间
            data[col] = (data[col] - col_min) / (col_max - col_min)

    # 保存归一化后的数据，不保留索引列
    data.to_csv(consistent_testing_info_normalized_file, index=False)

def append_features_to_attribute_file(project_dir, attribute_file, attribute_normalized_file):
    consistent_testing_info_file_path = join_path(project_dir, attribute_file)
    consistent_testing_info_normalized_file = join_path(project_dir, attribute_normalized_file)
    data = pandas.read_csv(consistent_testing_info_file_path)

    # 确定需要归一化的列（排除'model'和'LABEL'）
    columns_to_normalize = [col for col in data.columns if col not in ['model', 'LABEL']]

    # 对每列进行最小-最大归一化 (Min-Max Normalization)
    for col in columns_to_normalize:
        # 获取当前列的最小值和最大值
        col_min = data[col].min()
        col_max = data[col].max()

        # 处理极端情况：如果所有值都相同（避免除以0）
        if col_max == col_min:
            data[col] = 0.0  # 全部归一化为0
        else:
            # 归一化公式：(x - min) / (max - min)，将值缩放到[0, 1]区间
            data[col] = (data[col] - col_min) / (col_max - col_min)
    cols_to_drop = ["correctness_reflectability_0", "correctness_reflectability_1","correctness_reflectability_2","correctness_reflectability_3",
                    "correctness_reflectability_4","correctness_reflectability_5","correctness_reflectability_6","correctness_reflectability_7",
                    "correctness_reflectability_8","correctness_reflectability_9"]
    target_df = pandas.read_csv(consistent_testing_info_normalized_file)
    target_df = target_df.drop(columns=cols_to_drop, axis=1, errors='ignore')
    concatenated_df = pandas.concat([target_df, data.iloc[:,2:]], axis=1, ignore_index=False)
    # 保存归一化后的数据，不保留索引列
    concatenated_df.to_csv(consistent_testing_info_normalized_file, index=False)


def calculate_attributes(project_dir, label_file, attribute_temp_file,
                         attribute_normalized_file, FIELDS):
    if not os.path.isfile(join_path(project_dir, label_file)):
        return
    attribute_data = {}
    failing_variants = get_labeled_failing_variants(project_dir, label_file)
    system_stm_ids = get_all_stm_ids(project_dir)
    failed_executions_in_failing_products = get_failings_executions(project_dir, system_stm_ids,
                                                                    failing_variants)

    passed_executions_in_failing_products = get_passing_executions(project_dir, system_stm_ids,
                                                                   failing_variants)
    variants_and_labels = get_variants_and_labels(project_dir, label_file)
    passing_variants_stmts = get_stmts_id_in_passing_variants(project_dir, failing_variants)
    susp_in_passing_variants = {}
    susp_scores_in_system = ranking_suspicious_stmts(project_dir, failing_variants)
    for p_v in passing_variants_stmts:
        attribute_data[p_v] = {}
        susp_in_passing_variants[p_v] = check_suspicious_stmts_in_passing_variants(
            failed_executions_in_failing_products, passing_variants_stmts[p_v])
        var_dir = join_path(join_path(project_dir, "variants"), p_v)
        attribute_data[p_v][LABEL] = variants_and_labels[p_v][LABEL]
        attribute_data[p_v][DDU] = 1 - ddu(var_dir, variants_and_labels[p_v][LABEL])

        not_executed_susp_stmts = check_executed_susp_stmt_vs_susp_stmt_in_passing_variant(
            susp_in_passing_variants[p_v])
        attribute_data[p_v][
            code_coverage] = not_executed_susp_stmts
        passed_executions_in_passing_product = get_passing_executions_in_a_variant(project_dir,
                                                                                   system_stm_ids, p_v)

        attribute_data[p_v][
            incorrectness_verifiability] = check_incorrectness_verifiability(
            failed_executions_in_failing_products, passed_executions_in_passing_product,
            susp_in_passing_variants[p_v], 0.8)

        attribute_data[p_v][
            correctness_reflectability] = check_correctness_reflectability(
            failed_executions_in_failing_products, passed_executions_in_failing_products,
            passed_executions_in_passing_product, susp_in_passing_variants[p_v], 0.8)
        # bscp
        attribute_data[p_v][buggy_statement_containing_possibility] = check_total_susp_scores_in_passing_variant(
            susp_scores_in_system, passing_variants_stmts[p_v])

        dependencies_similarity = check_dependencies(join_path(project_dir, "variants"), p_v,
                                                     failed_executions_in_failing_products)

        attribute_data[p_v][bug_involving_statements] = dependencies_similarity["Both"]
    write_dict_to_file(join_path(project_dir, attribute_temp_file), attribute_data, FIELDS)
    normalization(FIELDS, project_dir, attribute_temp_file, attribute_normalized_file)
    os.remove(join_path(project_dir, attribute_temp_file))

def average_feature_by_label(labels, values, target):
    sum = 0
    count = 0
    for i in range(0, len(labels)):
        if labels[i] == target:
            count += 1
            sum += values[i]

    if count == 0:
        return sum
    return sum / count

def calculate_attributes_from_system_paths(system_paths):
    for system in system_paths:
        for bug in system_paths[system]:
            sys_path = system_paths[system][bug]
            if not os.path.exists(sys_path):
                continue
            mutated_projects = list_dir(sys_path)
            # pool = Pool(8)
            for mutated_project in mutated_projects:
                mu_project_path = join_path(sys_path, mutated_project)
                label_file_path = "variant_labels.csv"
                attributes_temp_path = join_path(mu_project_path, "attributes_temp-clap.csv")
                attributes_path = join_path(mu_project_path, "attributes-clap.csv")
                calculate_attributes(mu_project_path, label_file_path, attributes_temp_path, attributes_path, FIELDS = FIELDS)
                # calculate_attributes3(mu_project_path, label_file_path, attributes_temp_path, attributes_path)
                # if not os.path.isfile(attributes_path):
                #     print(attributes_path)
                # pool.apply_async(calculate_cc, (mu_project_path, label_file_path, attributes_temp_path, attributes_path))
                # calculate_cc(mu_project_path, label_file_path, attributes_temp_path, attributes_path)
                # if not os.path.isfile(attributes_path):
                #     print(attributes_path)
                    # FIELDS = ["model", "LABEL", "bscp", "bug_involving_statements", "code_coverage", "correctness_reflectability", "incorrectness_verifiability"]
                    #  calculate_attributes(mu_project_path, label_file_path, attributes_temp_path, attributes_path, FIELDS=FIELDS)
            # pool.close()
            # pool.join()

def filter_out_current_variants(failed_executions_in_failing_products, passed_executions_in_failing_products, v):
    filtered_failed = {k: v_data for k, v_data in failed_executions_in_failing_products.items() if k != v}
    filtered_passed = {k: v_data for k, v_data in passed_executions_in_failing_products.items() if k != v}
    return filtered_failed, filtered_passed


def normalized_dict(input_dict):
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

    return normalized_dict


def calculate_attributes3(project_dir, label_file, attribute_temp_file,  attribute_normalized_file):
    if not os.path.isfile(join_path(project_dir, label_file)):
        return
    attribute_data = {}
    failing_variants = get_labeled_failing_variants(project_dir, label_file)
    system_stm_ids = get_all_stm_ids(project_dir)
    failed_executions_in_failing_products = get_failings_executions(project_dir, system_stm_ids,
                                                                    failing_variants)
    passed_executions_in_failing_products = get_passing_executions(project_dir, system_stm_ids,
                                                                   failing_variants)
    variants_and_labels = get_variants_and_labels(project_dir, label_file)
    susp_in_variants = {}
    full_ranked_list = ranking_suspicious_stmts(project_dir, failing_variants)
    for v in system_stm_ids:
        attribute_data[v] = {}
        new_failed_executions_in_failing_products,new_passed_executions_in_failing_products = filter_out_current_variants(
            failed_executions_in_failing_products, passed_executions_in_failing_products, v)
        susp_in_variants[v] = check_suspicious_stmts_in_passing_variants(
            new_failed_executions_in_failing_products, system_stm_ids[v])
        attribute_data[v][LABEL] = variants_and_labels[v][LABEL]
        passed_executions_in_passing_product = get_passing_executions_in_a_variant(project_dir,
                                                                                   system_stm_ids, v)
        for index, susp_scores_in_system in enumerate(full_ranked_list):
            susp_scores_in_system = normalized_dict(susp_scores_in_system)
            attribute_data[v][code_coverage+"_"+str(index)] = calculate_cr(susp_in_variants[v], susp_scores_in_system,system_stm_ids[v])
            attribute_data[v][incorrectness_verifiability+"_"+str(index)] = check_similarity_score(
                new_failed_executions_in_failing_products, susp_in_variants[v], susp_scores_in_system
            )
            # bscp
            attribute_data[v][buggy_statement_containing_possibility+"_"+str(index)] = check_total_susp_scores_in_passing_variant(
                susp_scores_in_system, system_stm_ids[v]
            )
            # bug_involving_statement
            dependencies_similarity = check_dependencies_with_susp_score(join_path(project_dir, "variants"), v,
                                                                         new_failed_executions_in_failing_products,
                                                                         susp_scores_in_system)
            attribute_data[v][bug_involving_statements+"_"+str(index)] = dependencies_similarity["Both"]

        threshold_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        for threshold in threshold_list:
            # correctness_reflectability
            attribute_data[v][correctness_reflectability + "_" + str(threshold*10)] = check_correctness_reflectability(
                new_failed_executions_in_failing_products, new_passed_executions_in_failing_products,
                passed_executions_in_passing_product, susp_in_variants[v], threshold
            )
    attribute_data = organize_features(attribute_data)
    write_dict_to_file2(join_path(project_dir, attribute_temp_file), attribute_data)
    normalization2(project_dir, attribute_temp_file, attribute_normalized_file)
    os.remove(join_path(project_dir, attribute_temp_file))

def calculate_cc(project_dir, label_file, attribute_temp_file,  attribute_normalized_file):
    if not os.path.isfile(join_path(project_dir, label_file)):
        return
    attribute_data = {}
    failing_variants = get_labeled_failing_variants(project_dir, label_file)
    system_stm_ids = get_all_stm_ids(project_dir)
    failed_executions_in_failing_products = get_failings_executions(project_dir, system_stm_ids,
                                                                    failing_variants)
    passed_executions_in_failing_products = get_passing_executions(project_dir, system_stm_ids,
                                                                   failing_variants)
    variants_and_labels = get_variants_and_labels(project_dir, label_file)
    susp_in_variants = {}
    for v in system_stm_ids:
        attribute_data[v] = {}
        new_failed_executions_in_failing_products, new_passed_executions_in_failing_products = filter_out_current_variants(
            failed_executions_in_failing_products, passed_executions_in_failing_products, v)
        susp_in_variants[v] = check_suspicious_stmts_in_passing_variants(
            new_failed_executions_in_failing_products, system_stm_ids[v])
        attribute_data[v][LABEL] = variants_and_labels[v][LABEL]
        passed_executions_in_passing_product = get_passing_executions_in_a_variant(project_dir,
                                                                                   system_stm_ids, v)
        threshold_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        for i, threshold in enumerate(threshold_list):
            attribute_data[v][
                correctness_reflectability + "_" + str(i)] = check_correctness_reflectability(
                new_failed_executions_in_failing_products, new_passed_executions_in_failing_products,
                passed_executions_in_passing_product, susp_in_variants[v], threshold
            )
    attribute_data = organize_features(attribute_data)
    write_dict_to_file2(join_path(project_dir, attribute_temp_file), attribute_data)
    append_features_to_attribute_file(project_dir, attribute_temp_file, attribute_normalized_file)
    os.remove(join_path(project_dir, attribute_temp_file))

def organize_features(attribute_data):
    FEATURE_TYPE_ORDER = [
        "bscp",
        "bug_involving_statements",
        "code_coverage",
        "correctness_reflectability",
        "incorrectness_verifiability"
    ]

    reorganized_data = {}
    for model_name, raw_features in attribute_data.items():
        label = raw_features.get("LABEL")
        reorganized_features = {"LABEL": label} if label is not None else {}

        for feature_type in FEATURE_TYPE_ORDER:
            type_keys = [key for key in raw_features.keys() if key.startswith(f"{feature_type}_")]

            sorted_type_keys = sorted(
                type_keys,
                key=lambda x: int(x.rsplit("_", 1)[1])
            )

            for key in sorted_type_keys:
                reorganized_features[key] = raw_features[key]

        reorganized_data[model_name] = reorganized_features

    return reorganized_data

def do_slicing_statements(system_paths):
    pool = Pool(processes= 8)
    for system in system_paths:
        for bug in system_paths[system]:
            sys_path = system_paths[system][bug]
            if not os.path.exists(sys_path):
                continue
            mutated_projects = list_dir(sys_path)
            for mutated_project in mutated_projects:
                mu_project_path = join_path(sys_path, mutated_project)
                pool.apply_async(do_slice_all_statements, (mu_project_path,), error_callback=error_callback)
                # do_slice_all_statements(mu_project_path)
    pool.close()
    pool.join()


def error_callback(error):
    print(f"Error info: {error}")

