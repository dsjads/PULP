import logging
import os

import xml.etree.ElementTree as ET

from FileManager import get_variant_dir, get_test_coverage_dir, join_path, \
    get_failing_variants, get_spectrum_failed_coverage_file_name_with_version, \
    get_spectrum_passed_coverage_file_name_with_version, list_dir, get_variants_dir
from ranking.RankingManager import get_set_of_stms

from spectrum_manager.Spectrum_Expression import *
from ranking.Keywords import *


def features_ranking_multiple_bugs(buggy_statements, mutated_project_dir, failing_variants,
                                   FP_variants, add_more_tests, search_spaces, filter_coverage_rate, spectrum_expressions,
                                   spectrum_coverage_prefix=""):

    statements_sets = {SS_STMS_IN_F_PRODUCTS: get_set_of_stms(search_spaces[SS_STMS_IN_F_PRODUCTS]),
                       SS_SLICING: get_set_of_stms(search_spaces[SS_SLICING])}
    total_variants = 0

    features_info = {SS_STMS_IN_F_PRODUCTS: {}, SS_SLICING: {}}

    variants_dir = get_variants_dir(mutated_project_dir)
    total_passes = 0
    total_fails = 0
    for variant in list_dir(variants_dir):
        variant_dir = get_variant_dir(mutated_project_dir, variant)
        features_info, a_pass, a_fail = get_coverage_infor_of_variants(variant, variant_dir, failing_variants,  FP_variants, add_more_tests,features_info, statements_sets,
                                                       spectrum_coverage_prefix)
        total_passes += a_pass
        total_fails += a_fail

    # there are no passing variants with test coverage > threshold
    if total_passes == 0:
        return -2, -2, -2
    # there are no failing variants
    if total_fails == 0:
        return -2, -2, -2
    feature_based_rank = {}
    for spectrum_expression in spectrum_expressions:
        features_info[SS_STMS_IN_F_PRODUCTS] = features_suspiciousness_calculation(features_info[SS_STMS_IN_F_PRODUCTS], total_passes, total_fails, spectrum_expression)

        feature_based_rank[spectrum_expression] = {FB_RANK:{}, FB_TC_RANK:{}}
        for stm in buggy_statements:
            feature_rank, stm_rank = search_rank_worst_case(stm, features_info[SS_STMS_IN_F_PRODUCTS], features_info[SS_STMS_IN_F_PRODUCTS], spectrum_expression)
            feature_based_rank[spectrum_expression][FB_RANK][stm] = {}
            feature_based_rank[spectrum_expression][FB_RANK][stm][RANK] = stm_rank
            feature_based_rank[spectrum_expression][FB_RANK][stm][EXAM] = (stm_rank/ len(statements_sets[SS_STMS_IN_F_PRODUCTS])) * 100

        for stm in buggy_statements:
            if stm in statements_sets[SS_SLICING]:
                feature_rank, stm_rank = search_rank_worst_case(stm, features_info[SS_STMS_IN_F_PRODUCTS], features_info[SS_SLICING], spectrum_expression)
                feature_based_rank[spectrum_expression][FB_TC_RANK][stm] = {}
                feature_based_rank[spectrum_expression][FB_TC_RANK][stm][RANK] = stm_rank
                feature_based_rank[spectrum_expression][FB_TC_RANK][stm][EXAM] = (stm_rank/len(statements_sets[SS_SLICING])) * 100
            else:
                feature_based_rank[spectrum_expression][FB_TC_RANK] = feature_based_rank[spectrum_expression][FB_RANK]

    return feature_based_rank


def features_suspiciousness_calculation(features_info, total_passes, total_fails, spectrum_expression):
    score = spectrum_expression + "_score"
    for id in features_info.keys():
        features_info[id][score] = suspicious_score_by_sbfl_metric(spectrum_expression, len(features_info[id][VARIANTS_FAILED]),
                                                                   len(features_info[id][VARIANTS_PASSED]),
                                                                   total_fails,
                                                                   total_passes )

    return features_info


def get_coverage_infor_of_variants(variant, variant_dir, failing_variants,  FP_variants, add_more_tests,features_coverage_info, search_spaces,
                                   spectrum_coverage_prefix):
    test_coverage_dir = get_test_coverage_dir(variant_dir)

    spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_failed_coverage_file_dir = join_path(test_coverage_dir, spectrum_failed_file)

    spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_passed_coverage_file_dir = join_path(test_coverage_dir, spectrum_passed_file)
    a_pass = 0
    a_fail = 0

    if variant in failing_variants:

        if os.path.isfile(spectrum_failed_coverage_file_dir):
            features_coverage_info = read_coverage_info(variant, features_coverage_info, search_spaces,
                                                        spectrum_failed_coverage_file_dir, VARIANTS_FAILED)
            a_fail += 1

        if os.path.isfile(spectrum_passed_coverage_file_dir):
            features_coverage_info = read_coverage_info(variant, features_coverage_info,search_spaces,
                                                        spectrum_passed_coverage_file_dir, VARIANTS_FAILED)
    if add_more_tests and variant in FP_variants and os.path.isfile(spectrum_failed_coverage_file_dir):
        if os.path.isfile(spectrum_failed_coverage_file_dir):
            features_coverage_info = read_coverage_info(variant, features_coverage_info, search_spaces,
                                                        spectrum_failed_coverage_file_dir, VARIANTS_FAILED)
            a_fail += 1

        if os.path.isfile(spectrum_passed_coverage_file_dir):
            features_coverage_info = read_coverage_info(variant, features_coverage_info,search_spaces,
                                                        spectrum_passed_coverage_file_dir, VARIANTS_FAILED)
    if variant not in failing_variants and variant not in FP_variants:
        if os.path.isfile(spectrum_passed_coverage_file_dir):
            features_coverage_info = read_coverage_info(variant, features_coverage_info, search_spaces,
                                                        spectrum_passed_coverage_file_dir, VARIANTS_PASSED)
            a_pass += 1

    return features_coverage_info, a_pass, a_fail


def read_coverage_info(variant, coverage_info, search_spaces, coverage_file, kind_of_test_count):
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        project = root.find("project")

        for package in project:
            for file in package:
                for line in file:
                    id = line.get('featureClass')
                    for space in [SS_STMS_IN_F_PRODUCTS, SS_SLICING]:
                        if id not in coverage_info[space]:
                            coverage_info[space][id] = {}
                            coverage_info[space][id][VARIANTS_FAILED] = []
                            coverage_info[space][id][VARIANTS_PASSED] = []
                            coverage_info[space][id][STATEMENT_ID] = []
                        if int(line.get('count')) > 0 and variant not in coverage_info[space][id][kind_of_test_count]:
                            coverage_info[space][id][kind_of_test_count].append(variant)
                        stm_id = line.get('featureClass') + "." + line.get('featureLineNum')
                        if stm_id in search_spaces[space] and stm_id not in coverage_info[space][id][STATEMENT_ID]:
                            coverage_info[space][id][STATEMENT_ID].append(stm_id)
        return coverage_info
    except:
        logging.info("Exception when parsing %s", coverage_file)


def search_rank_worst_case(buggy_stm, features_info, space, spectrum_expression):
    ranking_score_type = spectrum_expression + "_score"
    score = 0
    for feature in features_info:
        if buggy_stm in features_info[feature][STATEMENT_ID]:
            score = features_info[feature][ranking_score_type]
            break

    feature_rank = 0
    stm_rank = 0
    for feature in features_info:
        if features_info[feature][ranking_score_type] >= score:
            feature_rank += 1
            stm_rank += len(space[feature][STATEMENT_ID])
    return feature_rank, stm_rank


def search_rank_best_case(buggy_stm, features_info, spectrum_expression):
    ranking_score_type = spectrum_expression + "_score"
    score = 0
    for feature in features_info:
        if buggy_stm in features_info[feature][STATEMENT_ID]:
            score = features_info[feature][ranking_score_type]
            break

    feature_rank = 0
    stm_rank = 0
    for feature in features_info:
        if features_info[feature][ranking_score_type] > score:
            feature_rank += 1
            stm_rank += len(features_info[feature][STATEMENT_ID])
    return feature_rank + 1, stm_rank + 1


def total_ranking_statements(features_info):
    stms = 0
    for feature in features_info:
        stms += len(features_info[feature][STATEMENT_ID])
    return stms
